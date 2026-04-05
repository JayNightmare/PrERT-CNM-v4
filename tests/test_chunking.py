from prert.chunking import chunk_record
from prert.extract.schema import ControlRecord, make_normalized_id, stable_hash


def test_chunking_respects_document_size_limit() -> None:
    text = "\n".join(["line-with-content" for _ in range(120)])
    native_id = "Article 1.1"
    normalized_id = make_normalized_id("GDPR", native_id)

    record = ControlRecord(
        record_id=stable_hash("seed")[:24],
        regulation="GDPR",
        source_document_id="gdpr-2016_679",
        source_path="sample",
        native_id=native_id,
        normalized_id=normalized_id,
        title="Title",
        text=text,
        hierarchy_path=["CHAPTER I", "Article 1", "1"],
    )

    chunks = chunk_record(record, max_document_bytes=128, max_lines_per_chunk=4)

    assert len(chunks) > 1
    assert all(len(c.text.encode("utf-8")) <= 128 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
