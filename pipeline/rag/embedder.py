import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.nomic import NomicEmbedding


class NomicEmbedder:
    """Creates and configures the Nomic embedding model used by the pipeline."""

    def __init__(self, model_name: str = "nomic-embed-text-v1") -> None:
        self._model_name = model_name
        self._embedding: BaseEmbedding | None = None

    def get_embedding(self) -> BaseEmbedding:
        if self._embedding is None:
            api_key = os.getenv("NOMIC_API_KEY")
            if not api_key:
                raise RuntimeError("NOMIC_API_KEY must be set before using the RAG pipeline.")

            self._embedding = NomicEmbedding(
                model_name=self._model_name,
                api_key=api_key,
            )
            Settings.embed_model = self._embedding

        return self._embedding
