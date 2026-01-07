from retriever.query.parse_query import parse_query


def test_parse_query_basic():
    parsed = parse_query("A red tie and a white shirt in a formal setting.")
    assert "red tie" in parsed.phrases or ("red" in parsed.colors and "tie" in parsed.garments)
    assert parsed.env in {"office", None}
    assert "red" in parsed.colors
    assert "shirt" in parsed.garments
    assert "tie" in parsed.items
