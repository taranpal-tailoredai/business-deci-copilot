from pathlib import Path

import faiss
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore


class FaissPolicyIndex:
    """Builds and reopens the locally persisted FAISS-backed LlamaIndex index."""

    def __init__(self, persist_dir: str | Path = ".data/faiss") -> None:
        self._persist_dir = Path(persist_dir)

    def create(self, nodes: list[BaseNode], embedding_dimension: int) -> VectorStoreIndex:
        if not nodes:
            raise ValueError("Cannot build a vector index from no document chunks.")

        vector_store = FaissVectorStore(
            faiss_index=faiss.IndexFlatL2(embedding_dimension)
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Keep the full tree for AutoMergingRetriever, but vector-search only leaves.
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex(get_leaf_nodes(nodes), storage_context=storage_context)
        self._persist(index)
        return index

    def load(self) -> VectorStoreIndex:
        vector_store_path = self._persist_dir / "default__vector_store.json"
        if not vector_store_path.is_file():
            raise FileNotFoundError(
                f"No persisted FAISS index found at {self._persist_dir}. Run ingest first."
            )

        vector_store = FaissVectorStore.from_persist_path(str(vector_store_path))
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self._persist_dir),
            vector_store=vector_store,
        )
        return load_index_from_storage(storage_context)

    def _persist(self, index: VectorStoreIndex) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(self._persist_dir))
