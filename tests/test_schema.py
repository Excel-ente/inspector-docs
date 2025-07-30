from docid.schema import Item, Summary, ZipResult


def test_schema_roundtrip():
    z = ZipResult(
        zip="a.zip",
        run_started_at="2020-01-01T00:00:00",
        items=[
            Item(
                file="a.png",
                page=1,
                class_name="otro",
                confidence=0.5,
                status="manual_review",
                fields={},
                ocr_chars=0,
            )
        ],
        summary=Summary(by_class={"otro": 1}, low_confidence=1),
    )
    data = z.json()
    z2 = ZipResult.parse_raw(data)
    assert z2.zip == "a.zip"
