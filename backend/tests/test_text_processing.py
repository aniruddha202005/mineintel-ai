import pytest

from app.services.text_processing import chunk_pages, clean_text


def test_clean_text_joins_hyphenated_line_breaks():
    assert clean_text("The ven-\ntilation fan failed") == "The ventilation fan failed"


def test_clean_text_turns_single_newlines_into_spaces_and_keeps_paragraphs():
    cleaned = clean_text("Line one\nline two\n\n\n\nSecond paragraph")
    assert cleaned == "Line one line two\n\nSecond paragraph"


def test_clean_text_collapses_whitespace_and_control_characters():
    assert clean_text("a   b\t\tc\x00d") == "a b c d"


def test_clean_text_empty_input():
    assert clean_text("  \n  ") == ""


def test_chunk_pages_keeps_page_numbers_and_global_index():
    pages = [(1, "First page text about a roof fall."), (2, "Second page text about ventilation.")]
    drafts = chunk_pages(pages, chunk_size=500, chunk_overlap=50)
    assert [d.page_number for d in drafts] == [1, 2]
    assert [d.chunk_index for d in drafts] == [0, 1]


def test_chunk_pages_respects_size_and_overlaps():
    sentence = "Rock bolts were installed along the haulage road. "
    drafts = chunk_pages([(1, sentence * 40)], chunk_size=200, chunk_overlap=50)
    assert len(drafts) > 1
    assert all(len(d.text) <= 200 for d in drafts)
    shared = set(drafts[0].text.split()) & set(drafts[1].text.split())
    assert shared


def test_chunk_pages_skips_blank_pages():
    drafts = chunk_pages([(1, "   "), (2, "Methane levels rose in the north panel.")], 500, 50)
    assert [d.page_number for d in drafts] == [2]


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_chunk_pages_rejects_bad_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([(1, "text")], size, overlap)
