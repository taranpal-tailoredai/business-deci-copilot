import pytest

pytest.importorskip("llama_index")
pytest.importorskip("faiss")
pytest.importorskip("llama_index.embeddings.nomic")
pytest.importorskip("llama_index.vector_stores.faiss")

from pipeline.rag.index import FaissPolicyIndex
from pipeline.rag.orchestrator import PolicyRagOrchestrator


def test_index_rejects_empty_chunks(tmp_path):
    with pytest.raises(ValueError, match="no document chunks"):
        FaissPolicyIndex(tmp_path).create([], embedding_dimension=768)


def test_retrieve_rejects_blank_question():
    with pytest.raises(ValueError, match="non-empty policy question"):
        PolicyRagOrchestrator().retrieve("   ")
