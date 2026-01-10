"""API routes for the Palia Q&A bot."""

import logging

from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import PlainTextResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


@router.get("/ask", response_class=PlainTextResponse)
@limiter.limit("30/minute")
async def ask_question(
    request: Request,
    q: str = Query(
        ...,
        description="The question to ask about Palia",
        min_length=1,
        max_length=500,
    ),
) -> str:
    """
    Nightbot-compatible Q&A endpoint.

    Returns a plain text response under 400 characters.

    Usage in Nightbot:
    !addcom !palia $(urlfetch https://your-domain.com/ask?q=$(querystring))

    Example queries:
    - /ask?q=what does Hassian like as a gift
    - /ask?q=where can I find iron ore
    - /ask?q=how do I catch a sturgeon
    """
    logger.info(f"Question received: {q[:100]}...")

    try:
        rag_engine = get_rag_engine()
        answer = rag_engine.query(q)
        return answer
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return "Sorry, I encountered an error. Please try again later."


@router.get("/ask/json")
@limiter.limit("30/minute")
async def ask_question_json(
    request: Request,
    q: str = Query(
        ...,
        description="The question to ask about Palia",
        min_length=1,
        max_length=500,
    ),
) -> dict:
    """
    JSON response endpoint for debugging and other integrations.

    Returns the answer along with source information.
    """
    logger.info(f"JSON question received: {q[:100]}...")

    try:
        rag_engine = get_rag_engine()
        answer = rag_engine.query(q)
        sources = rag_engine.get_sources(q)

        return {
            "question": q,
            "answer": answer,
            "sources": sources,
            "truncated": len(answer) >= 400,
        }
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Error processing question")


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    try:
        rag_engine = get_rag_engine()
        doc_count = rag_engine.vector_store.count()
        return {
            "status": "healthy",
            "documents_indexed": doc_count,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
