"""RAG engine for question answering."""

import hashlib
import logging
from functools import lru_cache

from cachetools import TTLCache
from openai import OpenAI

from app.config import get_settings
from app.core.vector_store import get_vector_store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant for the game Palia. Answer questions based ONLY on the provided context from the Palia wiki.

Rules:
1. Be concise - responses must be under 280 characters (a wiki link will be added after)
2. If the answer isn't in the context, say "I couldn't find that info."
3. Never make up information not in the context
4. Include specific details like locations, item names, or NPC names when relevant
5. For gift preferences, be specific about what the villager loves/likes/dislikes
6. Don't include citations or source references in your answer
7. Write in a friendly, helpful tone suitable for Twitch chat"""

USER_PROMPT_TEMPLATE = """Context from Palia Wiki:
{context}

Question: {question}

Provide a brief, helpful answer (under 280 characters):"""


class RAGEngine:
    """RAG engine for answering Palia questions."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.cache = TTLCache(
            maxsize=self.settings.cache_max_size,
            ttl=self.settings.cache_ttl,
        )

    def _cache_key(self, question: str) -> str:
        """Generate cache key from question."""
        normalized = question.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _build_context(self, chunks: list[dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for chunk in chunks:
            title = chunk["metadata"].get("title", "Unknown")
            section = chunk["metadata"].get("section", "")
            text = chunk["text"]

            # Include source info for context
            if section and section != "infobox":
                header = f"[{title} - {section}]"
            else:
                header = f"[{title}]"

            context_parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def _generate_response(self, question: str, context: str) -> str:
        """Generate a response using the LLM."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )

        response = self.client.chat.completions.create(
            model=self.settings.openai_chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.3,  # Low temperature for more consistent answers
        )

        return response.choices[0].message.content.strip()

    def _get_best_source_url(self, chunks: list[dict]) -> str | None:
        """Get the URL of the most relevant source."""
        for chunk in chunks:
            url = chunk["metadata"].get("url", "")
            if url:
                return url
        return None

    def query(self, question: str) -> str:
        """
        Answer a question about Palia.

        Args:
            question: The user's question

        Returns:
            Answer string with wiki link (max 400 chars for Nightbot)
        """
        # Check cache
        cache_key = self._cache_key(question)
        if cache_key in self.cache:
            logger.debug(f"Cache hit for question: {question[:50]}...")
            return self.cache[cache_key]

        # Retrieve relevant chunks
        chunks = self.vector_store.similarity_search(
            question,
            k=self.settings.retrieval_k,
        )

        if not chunks:
            return "I couldn't find that info in the wiki yet."

        # Build context and generate response
        context = self._build_context(chunks)
        answer = self._generate_response(question, context)

        # Get the most relevant wiki URL
        source_url = self._get_best_source_url(chunks)

        # Calculate max answer length to fit URL within 400 char limit
        max_len = self.settings.max_response_length
        if source_url:
            # Reserve space for " | " + URL
            url_space = len(source_url) + 3
            max_answer_len = max_len - url_space
        else:
            max_answer_len = max_len

        # Truncate answer if needed
        if len(answer) > max_answer_len:
            answer = answer[: max_answer_len - 3] + "..."

        # Append wiki link
        if source_url:
            answer = f"{answer} | {source_url}"

        # Cache the result
        self.cache[cache_key] = answer

        return answer

    def get_sources(self, question: str, k: int = 3) -> list[dict]:
        """
        Get source URLs for a question (for debugging/JSON endpoint).

        Args:
            question: The user's question
            k: Number of sources to return

        Returns:
            List of source info dicts
        """
        chunks = self.vector_store.similarity_search(question, k=k)
        sources = []
        seen_urls = set()

        for chunk in chunks:
            url = chunk["metadata"].get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": chunk["metadata"].get("title", "Unknown"),
                    "url": url,
                    "section": chunk["metadata"].get("section", ""),
                })

        return sources


# Singleton instance
_rag_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    """Get the RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
