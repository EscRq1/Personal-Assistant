from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import socket


HEADERS = {
    "User-Agent": (
        "MyPythonAssistant/1.0 "
        "(A student's pet project; "
        "contact: https://github.com/EscRq1)"
    )
}


def search_web(
    query: str,
    max_results: int = 5,
):
    """
    Search DuckDuckGo and return raw search results.
    """

    results = []

    with DDGS() as ddgs:
        for result in ddgs.text(
            query,
            max_results=max_results,
        ):
            results.append(result)

    return results


def fetch_page_text(
    url: str,
    max_chars: int = 3000,
):
    """
    Fetch and clean webpage text.
    """

    try:
        response = requests.get(
            url,
            timeout=5,
            headers=HEADERS,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for tag in soup(
            ["script", "style", "nav", "footer"]
        ):
            tag.decompose()

        text = " ".join(
            soup.stripped_strings
        )

        return text[:max_chars]

    except requests.RequestException:
        return ""


def is_online():
    """
    Cheap connectivity check.
    """

    try:
        socket.create_connection(
            ("8.8.8.8", 53),
            timeout=2,
        )
        return True

    except OSError:
        return False


def web_lookup(
    query: str,
    max_results: int = 3,
    page_chars: int = 8000,
    total_chars: int = 20000,
) -> str:
    """
    Raw web evidence collection.

    This function does NOT summarize the results.
    The Observer will do that later.
    """

    if not is_online():
        print(
            "[WEB] Offline; skipping search."
        )
        return ""

    print(
        f"[WEB] Searching for: {query}"
    )

    context_chunks = []

    try:
        search_results = search_web(
            query,
            max_results=max_results,
        )

        for result in search_results:

            url = (
                result.get("href")
                or result.get("url")
            )

            if not url:
                continue

            print(
                f"[WEB] Fetching: {url}"
            )

            text = fetch_page_text(
                url,
                max_chars=page_chars,
            )

            print(
                f"[WEB] Extracted: {len(text)} chars"
            )

            if not text:
                continue

            context_chunks.append(
                f"Source: {url}\n{text}"
            )

            if sum(
                len(chunk)
                for chunk in context_chunks
            ) >= total_chars:
                break

    except Exception as exc:
        print(
            f"[WEB] Search failed: {exc}"
        )
        return ""

    return "\n\n".join(
        context_chunks
    )