"""Wiki scraper for palia.wiki.gg."""

import gzip
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO

import requests
from bs4 import BeautifulSoup

from .page_parser import parse_wiki_page, WikiPage
from .chunker import create_chunks, Chunk

logger = logging.getLogger(__name__)

WIKI_BASE_URL = "https://palia.wiki.gg"
SITEMAP_INDEX_URL = "https://palia.wiki.gg/sitemaps/sitemap-index-palia_en.xml"

# Patterns to skip
SKIP_PATTERNS = [
    "/wiki/File:",
    "/wiki/Template:",
    "/wiki/Category:",
    "/wiki/Special:",
    "/wiki/User:",
    "/wiki/Talk:",
    "/wiki/Module:",
    "/wiki/MediaWiki:",
    "/wiki/Guide:",
    "?action=",
    "/wiki/Main_Page",
]

# Language suffixes to skip (keep only English)
LANGUAGE_SUFFIXES = ["/de", "/es", "/fr", "/it", "/ja", "/ko", "/pl", "/pt-br", "/ru", "/th", "/tr", "/uk", "/vi", "/zh-hans", "/zh-tw"]

# Priority pages to scrape first (most useful for Q&A)
PRIORITY_PATTERNS = [
    r"/wiki/[A-Z][a-z]+$",  # Simple page names (often characters/items)
    r"/wiki/Quests",
    r"/wiki/Skills",
    r"/wiki/Locations",
    r"/wiki/Gifting",
]

DEFAULT_HEADERS = {
    "User-Agent": "PaliaWikiBot/1.0 (Educational Project; Q&A Bot)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass
class ScraperConfig:
    """Configuration for the wiki scraper."""

    delay: float = 1.0  # Seconds between requests
    max_pages: int | None = None  # Limit for testing
    timeout: int = 30


class WikiScraper:
    """Scraper for the Palia wiki."""

    def __init__(self, config: ScraperConfig | None = None):
        self.config = config or ScraperConfig()
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def should_skip_url(self, url: str) -> bool:
        """Check if URL should be skipped."""
        for pattern in SKIP_PATTERNS:
            if pattern in url:
                return True
        # Skip non-English pages
        for suffix in LANGUAGE_SUFFIXES:
            if url.endswith(suffix):
                return True
        return False

    def _fetch_xml(self, url: str) -> ET.Element | None:
        """Fetch and parse an XML document (handles .gz compressed files)."""
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()

            content = response.content
            # Decompress if gzipped
            if url.endswith('.gz'):
                content = gzip.decompress(content)

            return ET.fromstring(content)
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
        except (ET.ParseError, gzip.BadGzipFile) as e:
            logger.error(f"Failed to parse XML from {url}: {e}")
            return None

    def get_sitemap_urls(self) -> list[str]:
        """Fetch all page URLs from the sitemap index."""
        logger.info(f"Fetching sitemap index from {SITEMAP_INDEX_URL}")

        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # First, get the sitemap index
        root = self._fetch_xml(SITEMAP_INDEX_URL)
        if root is None:
            return []

        # Check if this is a sitemap index (contains <sitemap> elements)
        # or a regular sitemap (contains <url> elements)
        sitemap_locs = root.findall(".//ns:sitemap/ns:loc", namespace)

        all_urls = []

        if sitemap_locs:
            # This is a sitemap index - fetch each child sitemap
            logger.info(f"Found sitemap index with {len(sitemap_locs)} sitemaps")
            for sitemap_loc in sitemap_locs:
                if sitemap_loc.text:
                    logger.info(f"Fetching sitemap: {sitemap_loc.text}")
                    sitemap_root = self._fetch_xml(sitemap_loc.text)
                    if sitemap_root is not None:
                        urls = [
                            loc.text
                            for loc in sitemap_root.findall(".//ns:loc", namespace)
                            if loc.text
                        ]
                        all_urls.extend(urls)
                    time.sleep(0.5)  # Be polite between sitemap fetches
        else:
            # This is a regular sitemap
            all_urls = [
                loc.text
                for loc in root.findall(".//ns:loc", namespace)
                if loc.text
            ]

        logger.info(f"Found {len(all_urls)} total URLs from sitemaps")
        return all_urls

    def fetch_page(self, url: str) -> str | None:
        """Fetch a single wiki page."""
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def scrape_page(self, url: str) -> WikiPage | None:
        """Scrape and parse a single wiki page."""
        html = self.fetch_page(url)
        if not html:
            return None

        try:
            return parse_wiki_page(html, url)
        except Exception as e:
            logger.warning(f"Failed to parse {url}: {e}")
            return None

    def scrape_all(self, progress_callback=None) -> list[Chunk]:
        """
        Scrape all wiki pages and return chunks.

        Args:
            progress_callback: Optional callback(current, total, url) for progress updates

        Returns:
            List of all text chunks with metadata
        """
        urls = self.get_sitemap_urls()

        # Filter out URLs to skip
        urls = [url for url in urls if not self.should_skip_url(url)]
        logger.info(f"Filtered to {len(urls)} URLs after removing skipped patterns")

        # Sort to prioritize important pages
        def priority_key(url):
            for i, pattern in enumerate(PRIORITY_PATTERNS):
                if re.search(pattern, url):
                    return i
            return len(PRIORITY_PATTERNS)

        urls.sort(key=priority_key)

        # Apply max_pages limit
        if self.config.max_pages:
            urls = urls[: self.config.max_pages]
            logger.info(f"Limited to {len(urls)} pages for testing")

        all_chunks = []
        for i, url in enumerate(urls):
            if progress_callback:
                progress_callback(i + 1, len(urls), url)

            page = self.scrape_page(url)
            if page:
                chunks = create_chunks(page)
                all_chunks.extend(chunks)
                logger.debug(f"Scraped {url}: {len(chunks)} chunks")

            # Rate limiting
            if i < len(urls) - 1:  # Don't sleep after last request
                time.sleep(self.config.delay)

        logger.info(f"Scraping complete: {len(all_chunks)} total chunks from {len(urls)} pages")
        return all_chunks


def scrape_single_page(url: str) -> list[Chunk]:
    """Convenience function to scrape a single page."""
    scraper = WikiScraper(ScraperConfig(delay=0))
    page = scraper.scrape_page(url)
    if page:
        return create_chunks(page)
    return []
