from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import BaseNode, Document


class PolicyChunker:
    """Builds parent-child policy nodes for precise retrieval with full context."""

    def __init__(self, chunk_sizes: list[int] | None = None) -> None:
        self._parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes or [1024, 512, 128]
        )

    def chunk(self, documents: list[Document]) -> list[BaseNode]:
        return self._parser.get_nodes_from_documents(documents)
