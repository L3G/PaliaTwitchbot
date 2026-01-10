"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.routes import router, limiter
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Palia Wiki Q&A Bot",
    description="A RAG-based Q&A bot for Palia using wiki.gg content",
    version="1.0.0",
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Nightbot and other services
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    settings = get_settings()
    logger.info(f"Starting Palia Q&A Bot on {settings.api_host}:{settings.api_port}")
    logger.info(f"Using model: {settings.openai_chat_model}")

    # Pre-initialize the RAG engine to avoid cold start on first request
    try:
        from app.core.rag_engine import get_rag_engine
        engine = get_rag_engine()
        doc_count = engine.vector_store.count()
        logger.info(f"Vector store initialized with {doc_count} documents")
    except Exception as e:
        logger.warning(f"Could not initialize RAG engine on startup: {e}")
        logger.warning("Make sure to run the scraper first: python -m scraper.run_scraper")


@app.get("/")
async def root():
    """Root endpoint with usage information."""
    return {
        "name": "Palia Wiki Q&A Bot",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "Plain text Q&A (Nightbot compatible)",
            "/ask/json": "JSON Q&A with sources",
            "/health": "Health check",
            "/docs": "API documentation",
        },
        "nightbot_setup": "!addcom !palia $(urlfetch https://your-domain.com/ask?q=$(querystring))",
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
