"""
Author: L. Saetta
Date Last Modified: 2026-10-26
License: MIT
Description: Reading text from external sources: web pages and plain-text URLs.
"""

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_MAX_BYTES = 2_000_000
USER_AGENT = "build-llm/0.1 (educational tokenizer demo)"

# Content of these HTML elements is not text for a human reader, or is
# site navigation rather than page content.
_SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg", "head", "nav"}

# These HTML elements hold the main content of a page. When a page has them,
# only their text is kept, which drops menus, sidebars, and footers.
_MAIN_CONTENT_TAGS = {"main", "article"}

# These HTML elements start a new line in the extracted text.
_BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
    "dt", "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "ol", "p", "pre", "section",
    "table", "td", "th", "tr", "ul",
}  # fmt: skip


class TextSourceError(Exception):
    """Raised when the text cannot be read from a source."""


@dataclass
class FetchedText:
    """Text read from a URL.

    Attributes:
        url: The URL that was read, after any redirects.
        title: The page title, or "" if not available.
        content_type: The media type of the response, for example "text/html".
        text: The extracted text.
    """

    url: str
    title: str
    content_type: str
    text: str


class _TextExtractor(HTMLParser):
    """HTML parser that collects the visible text and the page title.

    It collects two versions of the text: the text of the whole page, and the
    text inside the main content elements (see _MAIN_CONTENT_TAGS).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page_parts: list[str] = []
        self.main_parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._main_depth = 0
        self._in_title = False

    def _add(self, text: str) -> None:
        self.page_parts.append(text)
        if self._main_depth > 0:
            self.main_parts.append(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
        if tag in _MAIN_CONTENT_TAGS:
            self._main_depth += 1
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._add("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._add("\n")
        if tag in _MAIN_CONTENT_TAGS:
            self._main_depth = max(0, self._main_depth - 1)

    def handle_data(self, data: str) -> None:
        # The title is inside <head>, which is skipped as body text.
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0:
            self._add(data)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces, and keep at most one empty line in a row.

    Args:
        text: The text to normalize.

    Returns:
        The normalized text, without leading or trailing whitespace.
    """
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Extract the readable text and the title from an HTML page.

    If the page marks its main content with <main> or <article>, only that
    content is returned; otherwise the text of the whole page is returned.

    Args:
        html: The HTML source.

    Returns:
        A (title, text) pair.
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    title = normalize_whitespace(extractor.title)
    main_text = normalize_whitespace("".join(extractor.main_parts))
    page_text = normalize_whitespace("".join(extractor.page_parts))
    return title, main_text or page_text


def validate_url(url: str) -> None:
    """Check that a URL is an absolute http or https URL.

    Args:
        url: The URL to check.

    Raises:
        TextSourceError: If the URL is not valid.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise TextSourceError(
            f"Invalid URL '{url}': use an absolute http:// or https:// URL."
        )


def fetch_text_from_url(
    url: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> FetchedText:
    """Download a web page or a text file, and extract its text.

    HTML pages are converted to plain text; text/* responses are returned
    as they are. Other content types (images, PDF, ...) are rejected.

    Args:
        url: An absolute http or https URL.
        timeout: The maximum time to wait for the server, in seconds.
        max_bytes: The maximum size of the response to read.

    Returns:
        The extracted text.

    Raises:
        TextSourceError: If the URL is invalid, the download fails, or the
            content is not text or is too large.
    """
    validate_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            # Read one byte more than allowed, to detect content that is too large.
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as error:
        raise TextSourceError(
            f"The server returned HTTP {error.code} for '{url}'."
        ) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise TextSourceError(f"Cannot read '{url}': {error}") from error

    if len(body) > max_bytes:
        raise TextSourceError(
            f"The content of '{url}' is larger than {max_bytes:,} bytes."
        )

    is_html = content_type in ("text/html", "application/xhtml+xml")
    if not is_html and not content_type.startswith("text/"):
        raise TextSourceError(
            f"The content type of '{url}' is '{content_type}': only text and HTML "
            "pages are supported."
        )

    try:
        decoded = body.decode(charset, errors="replace")
    except LookupError:
        # The server declared an unknown charset: fall back to UTF-8.
        decoded = body.decode("utf-8", errors="replace")

    if is_html:
        title, text = html_to_text(decoded)
    else:
        title, text = "", decoded

    return FetchedText(url=final_url, title=title, content_type=content_type, text=text)
