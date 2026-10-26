"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: HTTP API (FastAPI) of the tokenizer demo, called by the web UI.

Run it from the repository root:

    python -m chap1.api

The interactive API documentation is then at http://127.0.0.1:8000/docs.
"""

import argparse
import sys
from dataclasses import asdict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from build_llm.text_sources import TextSourceError, fetch_text_from_url
from build_llm.tokenization import (
    DEFAULT_ENCODING,
    available_encodings,
    describe_encoding,
    get_encoding,
    tokenize,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

# Large texts produce very large JSON responses and slow down the UI.
MAX_TEXT_CHARACTERS = 200_000


class EncodingInfo(BaseModel):
    """An encoding that the user can choose."""

    name: str
    description: str


class EncodingsResponse(BaseModel):
    """The available encodings and the default one."""

    default: str
    encodings: list[EncodingInfo]


class TokenizeRequest(BaseModel):
    """A request to tokenize a text."""

    text: str = Field(max_length=MAX_TEXT_CHARACTERS)
    encoding: str = DEFAULT_ENCODING


class TokenModel(BaseModel):
    """One token, as defined by build_llm.tokenization.Token."""

    id: int
    text: str
    byte_values: list[int]
    is_partial_character: bool


class TokenizeResponse(BaseModel):
    """The result of a tokenization."""

    encoding: str
    vocabulary_size: int
    tokens: list[TokenModel]
    token_count: int
    character_count: int
    word_count: int
    characters_per_token: float


class FetchUrlRequest(BaseModel):
    """A request to read the text of a URL."""

    url: str


class FetchUrlResponse(BaseModel):
    """The text read from a URL."""

    url: str
    title: str
    content_type: str
    text: str
    truncated: bool


app = FastAPI(
    title="build-llm tokenizer demo",
    description="Tokenizes text with the BPE encodings of tiktoken.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    """Tell the caller that the API is running."""
    return {"status": "ok"}


@app.get("/encodings")
def list_encodings() -> EncodingsResponse:
    """Return the encodings that can be used for tokenization."""
    encodings = [
        EncodingInfo(name=name, description=describe_encoding(name))
        for name in available_encodings()
    ]
    return EncodingsResponse(default=DEFAULT_ENCODING, encodings=encodings)


@app.post("/tokenize")
def tokenize_text(request: TokenizeRequest) -> TokenizeResponse:
    """Tokenize a text with the requested encoding."""
    try:
        encoding = get_encoding(request.encoding)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    result = tokenize(request.text, encoding)
    return TokenizeResponse(**asdict(result))


@app.post("/fetch-url")
def fetch_url(request: FetchUrlRequest) -> FetchUrlResponse:
    """Read a web page or a text file, and return its text.

    Texts longer than the tokenization limit are truncated, so that the UI
    can tokenize them.
    """
    try:
        fetched = fetch_text_from_url(request.url)
    except TextSourceError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    truncated = len(fetched.text) > MAX_TEXT_CHARACTERS
    return FetchUrlResponse(
        url=fetched.url,
        title=fetched.title,
        content_type=fetched.content_type,
        text=fetched.text[:MAX_TEXT_CHARACTERS],
        truncated=truncated,
    )


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments of the API server."""
    parser = argparse.ArgumentParser(
        prog="python -m chap1.api",
        description="Start the HTTP API of the tokenizer demo.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"(default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"(default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart the server when the code changes (for development)",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Start the API server and serve requests until interrupted.

    Returns:
        The exit code.
    """
    arguments = parse_arguments(arguments)
    # With --reload, uvicorn needs the application as an import string.
    uvicorn.run(
        "chap1.api:app",
        host=arguments.host,
        port=arguments.port,
        reload=arguments.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
