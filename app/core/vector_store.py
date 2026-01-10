"""ChromaDB vector store operations."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.core.embeddings import get_embeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector store for wiki content."""

    def __init__(self):
        settings = get_settings()

        # Ensure persist directory exists
        persist_dir = Path(settings.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection_name = settings.chroma_collection_name
        self.embeddings = get_embeddings()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Palia wiki content for Q&A"},
        )

    def add_chunks(self, chunks: list, batch_size: int = 100) -> None:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of Chunk objects with text and metadata
            batch_size: Number of chunks to process at once
        """
        if not chunks:
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            texts = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            ids = [f"chunk_{i + j}" for j in range(len(batch))]

            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            logger.debug(f"Added batch {i // batch_size + 1}, total: {i + len(batch)}")

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        """
        Search for similar chunks.

        Args:
            query: The search query
            k: Number of results to return

        Returns:
            List of dicts with 'text', 'metadata', and 'distance'
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

        return formatted

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Palia wiki content for Q&A"},
        )
        logger.info("Cleared vector store")


# Singleton instance
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
