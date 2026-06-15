from app.ingestion.cleaning import clean_page_texts, normalize_document_text


def test_normalize_document_text_collapses_spacing_without_flattening_headings():
    text = "# 标题\t \n\n\n第一段  内容。\n第二段\u00a0内容。"

    cleaned = normalize_document_text(text)

    assert cleaned == "# 标题\n\n第一段 内容。\n第二段 内容。"


def test_clean_page_texts_removes_repeated_headers_and_footers():
    pages = [
        "Company Confidential\n\nRevenue grew in 2025.\n\nDraft Footer",
        "Company Confidential\n\nRisk factors changed.\n\nDraft Footer",
        "Company Confidential\n\nCash flow improved.\n\nDraft Footer",
    ]

    cleaned = clean_page_texts(pages)

    assert cleaned == [
        "Revenue grew in 2025.",
        "Risk factors changed.",
        "Cash flow improved.",
    ]


def test_clean_page_texts_keeps_single_page_content():
    cleaned = clean_page_texts(["Only one line of evidence."])

    assert cleaned == ["Only one line of evidence."]
