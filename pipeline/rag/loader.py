from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader


class PolicyDocumentLoader:
    """Loads policy PDFs while retaining LlamaIndex source metadata."""

    def __init__(self, input_dir: str | Path = "dataset/policy") -> None:
        self._input_dir = Path(input_dir)

    def load(self) -> list[Document]:
        if not self._input_dir.is_dir():
            raise FileNotFoundError(f"Policy directory does not exist: {self._input_dir}")

        return SimpleDirectoryReader(
            input_dir=str(self._input_dir),
            required_exts=[".pdf"],
        ).load_data()
