"""CLI script to run the wiki scraper and populate the vector database."""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.wiki_scraper import WikiScraper, ScraperConfig
from app.core.vector_store import VectorStore
from app.config import get_settings


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def progress_callback(current: int, total: int, url: str):
    """Print progress during scraping."""
    print(f"[{current}/{total}] Scraping: {url}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Palia wiki and populate vector database")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum number of pages to scrape (for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing database before scraping",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only scrape pages not already in the database",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    settings = get_settings()

    print("=" * 60)
    print("Palia Wiki Scraper")
    print("=" * 60)

    # Initialize vector store
    print(f"\nInitializing vector store at: {settings.chroma_persist_directory}")
    vector_store = VectorStore()

    if args.clear:
        print("Clearing existing database...")
        vector_store.clear()
        existing_urls = set()
    elif args.incremental:
        print("Incremental mode: fetching already indexed URLs...")
        existing_urls = vector_store.get_indexed_urls()
        print(f"Found {len(existing_urls)} already indexed URLs")
    else:
        existing_urls = set()

    # Configure and run scraper
    config = ScraperConfig(
        delay=args.delay,
        max_pages=args.max_pages,
    )

    print(f"\nStarting scrape with config:")
    print(f"  - Delay: {config.delay}s between requests")
    print(f"  - Max pages: {config.max_pages or 'unlimited'}")
    print(f"  - Incremental: {args.incremental}")
    print()

    scraper = WikiScraper(config)
    chunks = scraper.scrape_all(
        progress_callback=progress_callback,
        skip_urls=existing_urls if args.incremental else None,
    )

    if not chunks:
        if args.incremental:
            print("\nNo new pages to scrape. Database is up to date!")
            return 0
        print("\nNo chunks were scraped. Check the logs for errors.")
        return 1

    print(f"\nScraped {len(chunks)} chunks")
    print("Adding chunks to vector store...")

    # Add chunks to vector store
    vector_store.add_chunks(chunks)

    print(f"\nDone! Vector store now contains {vector_store.count()} documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
