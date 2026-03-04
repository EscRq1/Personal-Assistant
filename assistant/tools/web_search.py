from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import socket

# User-Agent for all HTTP requests
HEADERS = {
    "User-Agent": "MyPythonAssistant/1.0 (A student's pet project; contact: https://github.com/EscRq1)"
}

def search_web(query, max_results=5):
    """
    Uses DuckDuckGo to get search result links/text.
    DDGS does not accept headers in constructor.
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r)
    return results

def fetch_page_text(url, max_chars=3000):
    """
    Fetches page text from a URL, cleans it up, and limits length.
    Headers are applied here to be polite to sites like Wikipedia.
    """
    try:
        resp = requests.get(url, timeout=5, headers=HEADERS)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = " ".join(soup.stripped_strings)
        return text[:max_chars]

    except requests.RequestException:
        return ""


def is_online():
    try:
        # Try to connect to Google's public DNS on port 53
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False