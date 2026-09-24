from pathlib import Path

import numpy as np

from rag.cleaner import clean_text
from rag.chunker import chunk_text


def test_clean_text_removes_noise():
    input_text = "  This   is\n\n\n a test.  \n\nThis is another line.\n"
    cleaned = clean_text(input_text)
    assert "This is a test." in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_text_has_expected_shape():
    text = " ".join([f"word{i}" for i in range(2500)])
    chunks = chunk_text(text, document_id="doc-1", file_name="sample.pdf", chunk_size=700, overlap=100)
    assert chunks
    assert chunks[0]["metadata"]["document_id"] == "doc-1"
    assert chunks[0]["metadata"]["file_name"] == "sample.pdf"


def test_chunk_text_respects_overlap():
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, document_id="doc-2", file_name="sample.pdf", chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert chunks[0]["text"].split()[:10] == [f"word{i}" for i in range(10)]


def test_embedding_shapes_are_numeric():
    from rag.embeddings import embed_texts

    embeddings = embed_texts(["This is a legal judgment.", "This is a second sentence."])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
