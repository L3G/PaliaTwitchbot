"""Text chunking with overlap for RAG."""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .page_parser import WikiPage, format_infobox


@dataclass
class Chunk:
    """A text chunk with metadata for embedding."""

    text: str
    metadata: dict


def create_chunks(
    page: WikiPage,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Create overlapping text chunks from a wiki page.

    Args:
        page: Parsed wiki page
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks

    Returns:
        List of Chunk objects with text and metadata
    """
    chunks = []
    base_metadata = {
        "title": page.title,
        "url": page.url,
        "category": page.category,
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )

    # Infobox as a single chunk (usually short, structured data)
    if page.infobox:
        infobox_text = f"{page.title}\n\n{format_infobox(page.infobox)}"
        chunks.append(
            Chunk(
                text=infobox_text,
                metadata={**base_metadata, "section": "infobox"},
            )
        )

    # Chunk each section independently to preserve context
    for section in page.sections:
        # Prefix each chunk with title and section for context
        section_prefix = f"{page.title} - {section.heading}\n\n"

        # Split the section content
        section_chunks = splitter.split_text(section.content)

        for i, chunk_text in enumerate(section_chunks):
            # Add prefix to chunk for better retrieval context
            full_chunk_text = section_prefix + chunk_text

            chunks.append(
                Chunk(
                    text=full_chunk_text,
                    metadata={
                        **base_metadata,
                        "section": section.heading,
                        "chunk_index": i,
                    },
                )
            )

    # If no sections were found, chunk the full text
    if not page.sections and not page.infobox:
        full_chunks = splitter.split_text(page.full_text)
        for i, chunk_text in enumerate(full_chunks):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "section": "full_text",
                        "chunk_index": i,
                    },
                )
            )

    return chunks
