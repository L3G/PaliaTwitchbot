"""API routes for the Palia Q&A bot."""

import logging

import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, Query, Request, HTTPException, Path
from fastapi.responses import PlainTextResponse
from openai import OpenAI
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings
from app.core.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

# Zodiac sign name to number mapping
ZODIAC_SIGNS = {
    "aries": 1,
    "taurus": 2,
    "gemini": 3,
    "cancer": 4,
    "leo": 5,
    "virgo": 6,
    "libra": 7,
    "scorpio": 8,
    "sagittarius": 9,
    "capricorn": 10,
    "aquarius": 11,
    "pisces": 12,
}


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


@router.get("/horoscope/{sign}", response_class=PlainTextResponse)
@limiter.limit("30/minute")
async def get_horoscope(
    request: Request,
    sign: str = Path(..., description="Zodiac sign (e.g., aries, taurus, gemini)"),
) -> str:
    """
    Get today's horoscope for a zodiac sign, condensed to 350 characters.

    Usage in Nightbot:
    !addcom !horoscope $(urlfetch https://your-domain.com/horoscope/$(querystring))

    Example: /horoscope/aries
    """
    sign_lower = sign.lower().strip()

    if sign_lower not in ZODIAC_SIGNS:
        valid_signs = ", ".join(ZODIAC_SIGNS.keys())
        return f"Invalid sign '{sign}'. Valid signs: {valid_signs}"

    sign_number = ZODIAC_SIGNS[sign_lower]
    horoscope_url = f"https://www.horoscope.com/us/horoscopes/general/horoscope-general-daily-today.aspx?sign={sign_number}"

    try:
        # Fetch the horoscope page
        response = requests.get(horoscope_url, timeout=10)
        response.raise_for_status()

        # Parse the HTML
        soup = BeautifulSoup(response.text, "lxml")

        # Find the horoscope text (it's in the main-horoscope div)
        horoscope_div = soup.find("div", class_="main-horoscope")
        if not horoscope_div:
            return "Could not fetch horoscope. Please try again later."

        # Get the paragraph with the horoscope text
        horoscope_p = horoscope_div.find("p")
        if not horoscope_p:
            return "Could not parse horoscope. Please try again later."

        horoscope_text = horoscope_p.get_text(strip=True)

        # Remove the date prefix (e.g., "Jan 13, 2026 - ")
        if " - " in horoscope_text:
            horoscope_text = horoscope_text.split(" - ", 1)[1]

        # Use OpenAI to condense the horoscope
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

        condensed = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You condense horoscopes into brief, engaging summaries. Keep the essence and key advice. Max 280 characters. Don't use quotes.",
                },
                {
                    "role": "user",
                    "content": f"Condense this horoscope: {horoscope_text}",
                },
            ],
            max_tokens=100,
            temperature=0.7,
        )

        condensed_text = condensed.choices[0].message.content.strip()

        # Build the response with link
        sign_title = sign_lower.capitalize()
        link = f"https://www.horoscope.com/zodiac-signs/{sign_lower}"

        # Ensure total response fits in 400 chars
        max_text_len = 400 - len(link) - len(f"{sign_title}: ") - 3  # 3 for " | "
        if len(condensed_text) > max_text_len:
            condensed_text = condensed_text[: max_text_len - 3] + "..."

        return f"{sign_title}: {condensed_text} | {link}"

    except requests.RequestException as e:
        logger.error(f"Failed to fetch horoscope: {e}")
        return "Could not fetch horoscope. Please try again later."
    except Exception as e:
        logger.error(f"Error processing horoscope: {e}")
        return "Error processing horoscope. Please try again later."
