import json
from pathlib import Path
from typing import Any

from llama_index.core.schema import NodeWithScore


class RetrievalContextStore:
    """Persists the latest retrieval result for downstream answer generation."""

    def __init__(self, path: str | Path = ".data/faiss/context.json") -> None:
        self._path = Path(path)

    def save(self, query: str, nodes: list[NodeWithScore]) -> dict[str, Any]:
        context = {
            "query": query,
            "chunks": [
                {
                    "node_id": node.node.node_id,
                    "score": node.score,
                    "text": node.node.get_content(),
                    "metadata": node.node.metadata,
                }
                for node in nodes
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(context, indent=2, default=str),
            encoding="utf-8",
        )
        return context

    def load(self) -> dict[str, Any]:
        if not self._path.is_file():
            raise FileNotFoundError(
                f"No retrieval context found at {self._path}. Run retrieve first."
            )
        return json.loads(self._path.read_text(encoding="utf-8"))
