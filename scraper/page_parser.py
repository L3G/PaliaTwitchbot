"""HTML parsing and content extraction from wiki pages."""

import re
from dataclasses import dataclass, field
from bs4 import BeautifulSoup, Tag


@dataclass
class WikiSection:
    """A section of a wiki page."""

    heading: str
    content: str


@dataclass
class WikiPage:
    """Parsed content from a wiki page."""

    title: str
    url: str
    category: str = ""
    infobox: dict = field(default_factory=dict)
    sections: list[WikiSection] = field(default_factory=list)
    full_text: str = ""


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove edit links like [edit]
    text = re.sub(r"\[edit\]", "", text)
    # Strip leading/trailing whitespace
    return text.strip()


def extract_infobox(soup: BeautifulSoup) -> dict:
    """Extract structured data from the infobox sidebar."""
    infobox = {}

    # Try different infobox class patterns
    infobox_elem = soup.find("aside", class_="portable-infobox") or soup.find(
        "table", class_="infobox"
    )

    if not infobox_elem:
        return infobox

    # For portable-infobox (wiki.gg style)
    if infobox_elem.name == "aside":
        # Get title
        title_elem = infobox_elem.find("h2", class_="pi-title")
        if title_elem:
            infobox["name"] = clean_text(title_elem.get_text())

        # Get data items
        for item in infobox_elem.find_all("div", class_="pi-item"):
            label_elem = item.find("h3", class_="pi-data-label")
            value_elem = item.find("div", class_="pi-data-value")
            if label_elem and value_elem:
                label = clean_text(label_elem.get_text()).lower().replace(" ", "_")
                value = clean_text(value_elem.get_text())
                infobox[label] = value

    # For table-based infobox
    elif infobox_elem.name == "table":
        rows = infobox_elem.find_all("tr")
        for row in rows:
            header = row.find("th")
            data = row.find("td")
            if header and data:
                label = clean_text(header.get_text()).lower().replace(" ", "_")
                value = clean_text(data.get_text())
                infobox[label] = value

    return infobox


def extract_sections(soup: BeautifulSoup) -> list[WikiSection]:
    """Extract content sections from the page."""
    sections = []
    content_div = soup.find("div", class_="mw-parser-output")

    if not content_div:
        return sections

    current_heading = "Overview"
    current_content = []

    for element in content_div.children:
        if not isinstance(element, Tag):
            continue

        # Check for heading
        if element.name in ["h2", "h3"]:
            # Save previous section
            if current_content:
                content_text = clean_text(" ".join(current_content))
                if content_text:
                    sections.append(WikiSection(heading=current_heading, content=content_text))

            # Start new section
            headline = element.find("span", class_="mw-headline")
            current_heading = clean_text(headline.get_text()) if headline else clean_text(element.get_text())
            current_content = []

        # Skip navigation, tables of contents, etc.
        elif element.get("id") in ["toc", "catlinks", "mw-navigation"]:
            continue
        elif "navbox" in element.get("class", []):
            continue

        # Extract text from paragraphs, lists, tables
        elif element.name in ["p", "ul", "ol", "dl"]:
            text = clean_text(element.get_text())
            if text:
                current_content.append(text)

        elif element.name == "table" and "wikitable" in element.get("class", []):
            # Extract table data
            table_text = extract_table_text(element)
            if table_text:
                current_content.append(table_text)

    # Save last section
    if current_content:
        content_text = clean_text(" ".join(current_content))
        if content_text:
            sections.append(WikiSection(heading=current_heading, content=content_text))

    return sections


def extract_table_text(table: Tag) -> str:
    """Extract text content from a wiki table."""
    rows = []

    # Get headers
    headers = []
    header_row = table.find("tr")
    if header_row:
        for th in header_row.find_all("th"):
            headers.append(clean_text(th.get_text()))

    # Get data rows
    for row in table.find_all("tr")[1:]:  # Skip header row
        cells = row.find_all(["td", "th"])
        row_data = []
        for i, cell in enumerate(cells):
            cell_text = clean_text(cell.get_text())
            if headers and i < len(headers):
                row_data.append(f"{headers[i]}: {cell_text}")
            else:
                row_data.append(cell_text)
        if row_data:
            rows.append(", ".join(row_data))

    return "; ".join(rows)


def detect_category(url: str, soup: BeautifulSoup) -> str:
    """Detect the category of a wiki page from URL or content."""
    url_lower = url.lower()

    # Check URL patterns
    if "/villager" in url_lower or any(
        name in url_lower
        for name in [
            "ashura", "auni", "chayne", "delaila", "einar", "eshe",
            "hassian", "hodari", "jel", "jina", "kenyatta", "najuma",
            "nai'o", "reth", "sifuu", "tamala", "tish", "zeki"
        ]
    ):
        return "Villager"

    category_patterns = [
        ("fish", "Fish"),
        ("bug", "Bug"),
        ("recipe", "Recipe"),
        ("dish", "Dish"),
        ("quest", "Quest"),
        ("location", "Location"),
        ("skill", "Skill"),
        ("tool", "Tool"),
        ("furniture", "Furniture"),
        ("clothing", "Clothing"),
        ("seed", "Seed"),
        ("crop", "Crop"),
    ]

    for pattern, category in category_patterns:
        if pattern in url_lower:
            return category

    # Check categories in the page
    cat_links = soup.find("div", id="catlinks")
    if cat_links:
        cat_text = cat_links.get_text().lower()
        for pattern, category in category_patterns:
            if pattern in cat_text:
                return category

    return "General"


def parse_wiki_page(html: str, url: str) -> WikiPage:
    """Parse a wiki page HTML and extract structured content."""
    soup = BeautifulSoup(html, "lxml")

    # Get title
    title_elem = soup.find("h1", id="firstHeading") or soup.find("h1", class_="page-header__title")
    title = clean_text(title_elem.get_text()) if title_elem else "Unknown"

    # Extract components
    category = detect_category(url, soup)
    infobox = extract_infobox(soup)
    sections = extract_sections(soup)

    # Build full text
    full_text_parts = [title]
    if infobox:
        full_text_parts.append(format_infobox(infobox))
    for section in sections:
        full_text_parts.append(f"{section.heading}: {section.content}")
    full_text = "\n\n".join(full_text_parts)

    return WikiPage(
        title=title,
        url=url,
        category=category,
        infobox=infobox,
        sections=sections,
        full_text=full_text,
    )


def format_infobox(infobox: dict) -> str:
    """Format infobox data as readable text."""
    parts = []
    for key, value in infobox.items():
        # Convert snake_case to readable format
        readable_key = key.replace("_", " ").title()
        parts.append(f"{readable_key}: {value}")
    return "; ".join(parts)
