import requests
from bs4 import BeautifulSoup
from typing import Optional
from urllib.parse import urljoin, urlparse

DOCS_SOURCES = {
    "huggingface": {
        "url": "https://huggingface.co/docs/transformers/training",
        "base_path": "/docs/transformers/"
    },
    "langchain": {
        "url": "https://python.langchain.com/oss/python/langchain/overview",
        "base_path": "/oss/python/"
    },
}


def fetch_page(url: str):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["nav", "footer", "header", "script", "style", "code", "pre"]):
            tag.decompose()
        lines = soup.get_text(separator="\n", strip=True).split("\n")
        cleaned = "\n".join(l for l in lines if len(l) > 40)
        return cleaned, soup
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None, None


def get_doc_links(base_url: str, soup: BeautifulSoup, base_path: str) -> list[str]:
    links = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        if parsed.netloc == urlparse(base_url).netloc and parsed.path.startswith(base_path):
            links.add(href.split("#")[0])
    return list(links)


def fetch_docs(source: str, max_pages: int = 50) -> list[dict]:
    config = DOCS_SOURCES[source]
    base_url = config["url"]
    base_path = config["base_path"]
    print(f"Crawling {source} (up to {max_pages} pages)...")

    content, soup = fetch_page(base_url)
    if not soup:
        return []

    links = get_doc_links(base_url, soup, base_path)[:max_pages]
    print(f"  Found {len(links)} pages")

    results = []
    if content:
        results.append({"source": source, "url": base_url, "content": content})

    for i, url in enumerate(links):
        if url == base_url:
            continue
        text, _ = fetch_page(url)
        if text:
            results.append({"source": source, "url": url, "content": text})
        if i % 10 == 0:
            print(f"  {i}/{len(links)} pages fetched...")

    return results