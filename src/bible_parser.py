"""Parse the Bible Hub KJV PDF into structured verse records.

The source PDF has two consistent markers we exploit:

  * Chapter headers look like ``Genesis 1 \\nKJV  [Online]`` and start every
    chapter (multi-word / numbered books like ``1 Chronicles`` included).
  * Verse numbers are the digits immediately followed by U+202F (a narrow
    no-break space): ``1\\u202fIn the beginning God created...``. Cross-
    references such as ``(John 1:1-5)`` use ordinary punctuation, so they are
    never mistaken for verse markers.

parse_verses() walks the pages in order, carrying the current (book, chapter)
forward across continuation pages, and yields one dict per verse:
    {"book", "chapter", "verse", "ref", "text"}
"""
import re

from pypdf import PdfReader

# "<Book> <Chapter>\nKJV  [Online]" — non-greedy book capture up to the chapter number.
HEADER_RE = re.compile(r"([^\n]+?)\s+(\d+)\s*\nKJV\s*\[Online\]")

# A verse number sits on its own line, followed by a narrow no-break space
# (U+202F) before the verse text: "9\n And God said...".
VERSE_RE = re.compile("(\\d+)\\s* ")


def _clean(text):
    """Collapse whitespace and drop stray page artifacts from verse text."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_verses(pdf_path):
    """Yield structured verse dicts from the KJV PDF, in canonical order."""
    reader = PdfReader(pdf_path)
    book = None
    chapter = None

    for page in reader.pages:
        text = page.extract_text() or ""

        # Split the page on chapter headers so each span carries its own
        # (book, chapter). The first span inherits the previous page's chapter.
        last_end = 0
        spans = []  # (book, chapter, span_text)
        for m in HEADER_RE.finditer(text):
            spans.append((book, chapter, text[last_end:m.start()]))
            book, chapter = _clean(m.group(1)), int(m.group(2))
            last_end = m.end()
        spans.append((book, chapter, text[last_end:]))

        for span_book, span_chapter, span_text in spans:
            if span_book is None:  # before the very first header (title/TOC pages)
                continue
            # VERSE_RE.split -> [pre, num1, text1, num2, text2, ...]
            parts = VERSE_RE.split(span_text)
            for i in range(1, len(parts) - 1, 2):
                verse = int(parts[i])
                body = _clean(parts[i + 1])
                if not body:
                    continue
                yield {
                    "book": span_book,
                    "chapter": span_chapter,
                    "verse": verse,
                    "ref": f"{span_book} {span_chapter}:{verse}",
                    "text": body,
                }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/bible.pdf"
    verses = list(parse_verses(path))
    print(f"Parsed {len(verses)} verses")
    for v in verses[:3]:
        print(f"  {v['ref']}: {v['text'][:60]}")
    print("  ...")
    for v in verses[-3:]:
        print(f"  {v['ref']}: {v['text'][:60]}")
