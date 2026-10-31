"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Convert Standard Ebooks source archives into clean training text.
"""

import io
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import PurePosixPath

NAMESPACES = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "html": "http://www.w3.org/1999/xhtml",
}
EPUB_TYPE = "{http://www.idpf.org/2007/ops}type"
BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "tr"}
EXCLUDED_TYPES = {"noteref", "backlink", "footnote", "endnote", "endnotes", "rearnote"}


@dataclass
class BookText:
    """A converted book and its provenance.

    Attributes:
        title: Title from the EPUB metadata.
        author: Author names from the EPUB metadata.
        source: Canonical Standard Ebooks edition URL.
        text: UTF-8-ready plain text, including the title and author.
        chapters: Source paths retained in EPUB reading order.
    """

    title: str
    author: str
    source: str
    text: str
    chapters: list[str]


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Gutenberg header and footer when paired boundary markers exist.

    Args:
        text: Plain book text, possibly wrapped in Gutenberg metadata.

    Returns:
        Text between the START and END markers, or unchanged if neither exists.

    Raises:
        ValueError: If markers are incomplete or out of order.
    """
    start = re.search(
        r"^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}[^\n]*$",
        text,
        re.M | re.I,
    )
    end = re.search(
        r"^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}[^\n]*$",
        text,
        re.M | re.I,
    )
    if start is None and end is None:
        return text
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("Incomplete or reversed Gutenberg boundary markers.")
    return text[start.end() : end.start()].strip()


def _element_text(element: ET.Element) -> str:
    """Render blocks and line breaks, omitting note text and reference numbers."""
    tag = element.tag.rsplit("}", 1)[-1]
    types = set(element.get(EPUB_TYPE, "").split())
    if tag in {"script", "style", "svg"} or types & EXCLUDED_TYPES:
        return ""
    if tag == "br":
        return "\n"
    parts = [element.text or ""]
    for child in element:
        parts.extend((_element_text(child), child.tail or ""))
    text = "".join(parts)
    return f"\n\n{text}\n\n" if tag in BLOCK_TAGS else text


def _clean_text(text: str) -> str:
    """Remove formatting-only characters and normalize paragraph whitespace."""
    text = text.translate({ord(char): None for char in "\u2060\u200b\ufeff\u00ad"})
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def convert_standard_ebook(  # pylint: disable=too-many-locals
    archive: bytes,
) -> BookText:
    """Convert bodymatter XHTML in the source archive's EPUB spine order.

    The archive is read in memory without extracting any files to disk.
    Front/back matter, endnotes, and note references are excluded.

    Args:
        archive: A gzip-compressed tar archive from a Standard Ebooks repository.

    Returns:
        Clean text and the metadata and chapter paths needed for attribution.

    Raises:
        ValueError: If EPUB metadata or bodymatter is missing.
        tarfile.TarError: If the source archive is invalid.
        ET.ParseError: If an EPUB XML document is invalid.
    """
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        members = {member.name: member for member in source if member.isfile()}
        packages = [name for name in members if name.endswith("/src/epub/content.opf")]
        if len(packages) != 1:
            raise ValueError("Expected one EPUB content.opf in the source archive.")
        package_path = PurePosixPath(packages[0])

        def read_member(path: str) -> bytes:
            member = source.extractfile(members[path])
            if member is None:
                raise ValueError(f"Missing EPUB file: {path}")
            return member.read()

        package = ET.fromstring(read_member(str(package_path)))
        metadata = package.find("opf:metadata", NAMESPACES)
        if metadata is None:
            raise ValueError("Missing EPUB metadata.")
        title = metadata.findtext("dc:title", namespaces=NAMESPACES) or ""
        author = ", ".join(
            node.text or "" for node in metadata.findall("dc:creator", NAMESPACES)
        )
        url = metadata.findtext("dc:identifier", namespaces=NAMESPACES) or ""
        manifest = {
            item.attrib["id"]: item.attrib["href"]
            for item in package.findall("opf:manifest/opf:item", NAMESPACES)
        }
        chapters, texts = [], []
        for item in package.findall("opf:spine/opf:itemref", NAMESPACES):
            href = manifest[item.attrib["idref"]]
            document = ET.fromstring(read_member(str(package_path.parent / href)))
            body = document.find("html:body", NAMESPACES)
            if body is None or "bodymatter" not in body.get(EPUB_TYPE, "").split():
                continue
            text = _clean_text(_element_text(body))
            if text:
                chapters.append(f"src/epub/{href}")
                texts.append(text)
    if not title or not author or not texts:
        raise ValueError("The source must contain a title, author, and bodymatter.")
    body_text = strip_gutenberg_boilerplate("\n\n".join(texts))
    return BookText(
        title, author, url, f"{title}\n\nby {author}\n\n{body_text}\n", chapters
    )
