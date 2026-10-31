"""
Author: L. Saetta
Date Last Modified: 2026-10-31
License: MIT
Description: Offline tests of ebook extraction, cleaning, and corpus restoration.
"""

import hashlib
import io
import tarfile

import pytest

from build_llm.ebooks import convert_standard_ebook, strip_gutenberg_boilerplate
from chap4 import prepare_books


def make_archive() -> bytes:
    """Build a tiny source archive whose spine differs from filename order."""
    package = """<package xmlns="http://www.idpf.org/2007/opf">
      <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Example</dc:title><dc:creator>An Author</dc:creator>
        <dc:identifier>https://standardebooks.org/ebooks/example</dc:identifier>
      </metadata>
      <manifest>
        <item id="front" href="text/front.xhtml"/>
        <item id="one" href="text/one.xhtml"/>
        <item id="two" href="text/two.xhtml"/>
        <item id="notes" href="text/notes.xhtml"/>
      </manifest>
      <spine><itemref idref="front"/><itemref idref="two"/>
        <itemref idref="one"/><itemref idref="notes"/></spine>
    </package>"""
    documents = {
        "front": ("frontmatter", "<p>Publisher boilerplate</p>"),
        "one": ("bodymatter", "<h2>First chapter</h2><p>Last paragraph.</p>"),
        "two": (
            "bodymatter",
            """<h2>Second chapter</h2>
            <p>Some\u00a0<em>good</em> words\u2060—here<a epub:type="noteref">1</a>!</p>
            <p>A verse<br/>Another verse</p>
            <aside epub:type="footnote"><p>Hidden footnote</p></aside>""",
        ),
        "notes": ("backmatter", "<p>Hidden endnotes</p>"),
    }
    files = {"src/epub/content.opf": package}
    for name, (kind, content) in documents.items():
        files[f"src/epub/text/{name}.xhtml"] = f"""<html
          xmlns="http://www.w3.org/1999/xhtml"
          xmlns:epub="http://www.idpf.org/2007/ops">
          <head><title>Not body text</title></head>
          <body epub:type="{kind}">{content}</body></html>"""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in files.items():
            data = text.encode("utf-8")
            member = tarfile.TarInfo("example-revision/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def test_convert_keeps_body_in_spine_order():
    """Headings, inline text, and verse survive; editorial material is omitted."""
    book = convert_standard_ebook(make_archive())
    assert book.text.startswith("Example\n\nby An Author\n\nSecond chapter")
    assert book.chapters == ["src/epub/text/two.xhtml", "src/epub/text/one.xhtml"]
    assert "Some good words—here!" in book.text
    assert "A verse\nAnother verse" in book.text
    assert "Last paragraph." in book.text
    assert all(
        text not in book.text
        for text in ("Publisher", "Hidden", "Not body", "\u2060", "\u00a0")
    )


@pytest.mark.parametrize("article", ["THE", "THIS"])
def test_gutenberg_boundaries(article):
    """Only genuine paired markers remove Gutenberg metadata, on both sides."""
    text = (
        f"Header\n*** START OF {article} PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
        f"Story\n*** END OF {article} PROJECT GUTENBERG EBOOK EXAMPLE ***\nLicense"
    )
    assert strip_gutenberg_boilerplate(text) == "Story"
    assert (
        strip_gutenberg_boilerplate("Gutenberg appears in the story.")
        == "Gutenberg appears in the story."
    )


def test_incomplete_gutenberg_markers_fail():
    """An incomplete wrapper is rejected rather than silently leaving metadata."""
    with pytest.raises(ValueError, match="Incomplete"):
        strip_gutenberg_boilerplate(
            "*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nStory"
        )


def test_missing_epub_package_fails():
    """An archive without an EPUB package is rejected."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz"):
        pass
    with pytest.raises(ValueError, match="content.opf"):
        convert_standard_ebook(buffer.getvalue())


def test_prepare_book_checks_hash_and_preserves_modified_files(tmp_path, monkeypatch):
    """Restoration is deterministic and never overwrites an edited local book."""
    archive = make_archive()
    data = convert_standard_ebook(archive).text.encode("utf-8")
    entry = {
        "filename": "example.txt",
        "repository": "author_example",
        "revision": "a" * 40,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    monkeypatch.setattr(prepare_books, "BOOKS_DIR", tmp_path)
    monkeypatch.setattr(prepare_books, "download", lambda _url: archive)
    prepare_books.prepare_book(entry)
    path = tmp_path / "example.txt"
    assert path.read_bytes() == data
    path.write_text("User edits", encoding="utf-8")
    with pytest.raises(ValueError, match="existing file preserved"):
        prepare_books.prepare_book(entry)
    assert path.read_text(encoding="utf-8") == "User edits"


def test_prepare_book_rejects_changed_conversion(tmp_path, monkeypatch):
    """An unexpected source never becomes a training file."""
    monkeypatch.setattr(prepare_books, "BOOKS_DIR", tmp_path)
    monkeypatch.setattr(prepare_books, "download", lambda _url: make_archive())
    entry = {
        "filename": "example.txt",
        "repository": "author_example",
        "revision": "a" * 40,
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="Converted checksum mismatch"):
        prepare_books.prepare_book(entry)
    assert not (tmp_path / "example.txt").exists()
