"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Tests of the chap1.api HTTP API, with FastAPI's test client.
"""

import pytest
from fastapi.testclient import TestClient

from build_llm.text_sources import FetchedText, TextSourceError
from chap1 import api
from tests.conftest import HELLO


@pytest.fixture(name="client")
def fixture_client() -> TestClient:
    """A client that calls the API without starting a server."""
    return TestClient(api.app)


@pytest.fixture(name="tiny_api_encoding")
def fixture_tiny_api_encoding(monkeypatch, tiny_encoding):
    """Make the API use the tiny offline encoding for known names."""

    def get_encoding(name):
        if name not in api.available_encodings():
            raise ValueError(f"Unknown encoding '{name}'")
        return tiny_encoding

    monkeypatch.setattr(api, "get_encoding", get_encoding)


def test_health(client):
    """The health endpoint answers ok."""
    assert client.get("/health").json() == {"status": "ok"}


def test_encodings(client):
    """The encodings endpoint lists gpt2 as the default."""
    body = client.get("/encodings").json()

    assert body["default"] == "gpt2"
    assert "gpt2" in [encoding["name"] for encoding in body["encodings"]]


@pytest.mark.usefixtures("tiny_api_encoding")
def test_tokenize(client):
    """A text is tokenized and returned with its statistics."""
    response = client.post("/tokenize", json={"text": "hello", "encoding": "gpt2"})

    body = response.json()
    assert response.status_code == 200
    assert body["token_count"] == 1
    assert body["tokens"][0] == {
        "id": HELLO,
        "text": "hello",
        "byte_values": list(b"hello"),
        "is_partial_character": False,
    }


@pytest.mark.usefixtures("tiny_api_encoding")
def test_tokenize_unknown_encoding(client):
    """An unknown encoding gives HTTP 400."""
    response = client.post("/tokenize", json={"text": "hi", "encoding": "nope"})

    assert response.status_code == 400


def test_tokenize_text_too_long(client):
    """A text above the limit is rejected by validation (HTTP 422)."""
    text = "a" * (api.MAX_TEXT_CHARACTERS + 1)

    response = client.post("/tokenize", json={"text": text})

    assert response.status_code == 422


def test_fetch_url(monkeypatch, client):
    """The text of a URL is returned, truncated to the tokenization limit."""
    long_text = "a" * (api.MAX_TEXT_CHARACTERS + 5)
    monkeypatch.setattr(
        api,
        "fetch_text_from_url",
        lambda url: FetchedText(url, "Title", "text/html", long_text),
    )

    body = client.post("/fetch-url", json={"url": "https://x.test/"}).json()

    assert body["title"] == "Title"
    assert body["truncated"] is True
    assert len(body["text"]) == api.MAX_TEXT_CHARACTERS


def test_fetch_url_error(monkeypatch, client):
    """A URL that cannot be read gives HTTP 422 with the reason."""

    def failing_fetch(url):
        raise TextSourceError(f"Cannot read '{url}'")

    monkeypatch.setattr(api, "fetch_text_from_url", failing_fetch)

    response = client.post("/fetch-url", json={"url": "https://x.test/"})

    assert response.status_code == 422
    assert "Cannot read" in response.json()["detail"]
