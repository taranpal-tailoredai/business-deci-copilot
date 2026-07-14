from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.retrievers import AutoMergingRetriever

from .chunker import PolicyChunker
from .context import RetrievalContextStore
from .embedder import NomicEmbedder
from .index import FaissPolicyIndex
from .loader import PolicyDocumentLoader


class PolicyRagOrchestrator:
    """Coordinates hierarchical indexing and retrieval of policy context."""

    def __init__(
        self,
        policy_dir: str | Path = "dataset/policy",
        persist_dir: str | Path = ".data/faiss",
        model_name: str = "nomic-embed-text-v1",
        chunk_sizes: list[int] | None = None,
    ) -> None:
        self._loader = PolicyDocumentLoader(policy_dir)
        self._embedder = NomicEmbedder(model_name)
        self._index = FaissPolicyIndex(persist_dir)
        self._chunk_sizes = chunk_sizes
        self._context_store = RetrievalContextStore(Path(persist_dir) / "context.json")

    def ingest(self, documents: list[Document] | None = None) -> int:
        embedding = self._embedder.get_embedding()
        chunks = PolicyChunker(self._chunk_sizes).chunk(documents or self._loader.load())
        dimension = len(embedding.get_text_embedding("embedding dimension probe"))
        self._index.create(chunks, embedding_dimension=dimension)
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 3) -> dict[str, Any]:
        """Return and persist the highest-ranked, auto-merged policy chunks."""
        if not question.strip():
            raise ValueError("A non-empty policy question is required.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self._embedder.get_embedding()
        index = self._index.load()
        base_retriever = index.as_retriever(similarity_top_k=max(top_k * 4, 12))
        nodes = AutoMergingRetriever(
            base_retriever,
            index.storage_context,
            verbose=False,
        ).retrieve(question)
        return self._context_store.save(question, nodes[:top_k])

    def get_context(self) -> dict[str, Any]:
        return self._context_store.load()
