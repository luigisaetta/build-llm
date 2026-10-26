"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Tests of build_llm.text_sources, with a fake HTTP response.
"""

import urllib.error
from email.message import Message

import pytest

from build_llm import text_sources
from build_llm.text_sources import (
    TextSourceError,
    fetch_text_from_url,
    html_to_text,
    normalize_whitespace,
    validate_url,
)

SAMPLE_HTML = """
<html>
  <head><title> A   sample page </title><style>p { color: red; }</style></head>
  <body>
    <h1>Title</h1>
    <p>First   paragraph &amp; more.</p>
    <script>console.log("not text");</script>
    <ul><li>one</li><li>two</li></ul>
  </body>
</html>
"""


class FakeResponse:
    """A minimal stand-in for the response returned by urllib's urlopen."""

    def __init__(self, body: bytes, content_type: str, url: str = "https://x.test/"):
        self.body = body
        self.url = url
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def geturl(self) -> str:
        """Return the final URL."""
        return self.url

    def read(self, size: int = -1) -> bytes:
        """Return the body, up to size bytes."""
        return self.body if size < 0 else self.body[:size]

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def fake_urlopen(response):
    """Return a replacement for urlopen that always gives the response."""

    def urlopen(request, timeout):  # pylint: disable=unused-argument
        return response

    return urlopen


def test_html_to_text_keeps_only_visible_text():
    """Scripts and styles are dropped; blocks become lines; title is kept."""
    title, text = html_to_text(SAMPLE_HTML)

    assert title == "A sample page"
    assert text == "Title\n\nFirst paragraph & more.\n\none\n\ntwo"


def test_html_to_text_prefers_main_content():
    """With a <main> element, menus and footers outside it are dropped."""
    html = (
        "<body><nav>Menu</nav><header>Site name</header>"
        "<main><h1>Article</h1><p>Body text.</p></main>"
        "<footer>Copyright</footer></body>"
    )

    _, text = html_to_text(html)

    assert text == "Article\n\nBody text."


def test_html_to_text_skips_navigation():
    """Without a main element, the whole page is kept, except navigation."""
    _, text = html_to_text("<body><nav>Menu</nav><p>Hello</p></body>")

    assert text == "Hello"


def test_normalize_whitespace():
    """Spaces collapse and at most one empty line is kept."""
    assert normalize_whitespace("  a   b \n\n\n\n c\t ") == "a b\n\nc"


@pytest.mark.parametrize("url", ["ftp://x.test/file", "example.com", "https://"])
def test_validate_url_rejects_invalid_urls(url):
    """Only absolute http and https URLs are accepted."""
    with pytest.raises(TextSourceError):
        validate_url(url)


def test_fetch_html_page(monkeypatch):
    """An HTML page is converted to text."""
    response = FakeResponse(SAMPLE_HTML.encode(), "text/html; charset=utf-8")
    monkeypatch.setattr(text_sources.urllib.request, "urlopen", fake_urlopen(response))

    fetched = fetch_text_from_url("https://x.test/")

    assert fetched.title == "A sample page"
    assert fetched.content_type == "text/html"
    assert fetched.text.startswith("Title")


def test_fetch_plain_text_with_declared_charset(monkeypatch):
    """Plain text is returned as it is, decoded with the declared charset."""
    response = FakeResponse("perché".encode("latin-1"), "text/plain; charset=latin-1")
    monkeypatch.setattr(text_sources.urllib.request, "urlopen", fake_urlopen(response))

    fetched = fetch_text_from_url("https://x.test/file.txt")

    assert fetched.text == "perché"
    assert fetched.title == ""


def test_fetch_rejects_binary_content(monkeypatch):
    """Content that is not text, such as an image, is rejected."""
    response = FakeResponse(b"\x89PNG", "image/png")
    monkeypatch.setattr(text_sources.urllib.request, "urlopen", fake_urlopen(response))

    with pytest.raises(TextSourceError, match="image/png"):
        fetch_text_from_url("https://x.test/image.png")


def test_fetch_rejects_content_too_large(monkeypatch):
    """Content larger than max_bytes is rejected."""
    response = FakeResponse(b"a" * 11, "text/plain")
    monkeypatch.setattr(text_sources.urllib.request, "urlopen", fake_urlopen(response))

    with pytest.raises(TextSourceError, match="larger than"):
        fetch_text_from_url("https://x.test/big.txt", max_bytes=10)


def test_fetch_reports_http_errors(monkeypatch):
    """An HTTP error becomes a TextSourceError with the status code."""

    def failing_urlopen(request, timeout):  # pylint: disable=unused-argument
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", Message(), None
        )

    monkeypatch.setattr(text_sources.urllib.request, "urlopen", failing_urlopen)

    with pytest.raises(TextSourceError, match="HTTP 404"):
        fetch_text_from_url("https://x.test/missing")
