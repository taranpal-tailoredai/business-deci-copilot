"""Build the local policy index and save its top three retrieved chunks."""

from dotenv import load_dotenv

from pipeline.rag.orchestrator import PolicyRagOrchestrator


def main() -> None:
    load_dotenv()
    rag = PolicyRagOrchestrator()
    chunk_count = rag.ingest()
    context = rag.retrieve("What policy information is available?")

    if not context["chunks"]:
        raise RuntimeError("The policy index was built but returned no source nodes.")

    print(f"Indexed {chunk_count} chunks and saved {len(context['chunks'])} chunks to .data/faiss/context.json.")


if __name__ == "__main__":
    main()
