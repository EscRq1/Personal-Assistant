from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import socket
import ipaddress
from urllib.parse import urlparse


HEADERS = {
    "User-Agent": (
        "MyPythonAssistant/1.0 "
        "(A student's pet project; "
        "contact: https://github.com/EscRq1)"
    )
}

# ------------------------------------------------------------
# Safety / resource limits
# ------------------------------------------------------------

REQUEST_TIMEOUT = 5
CONNECTIVITY_TIMEOUT = 2

MAX_RESULTS = 5
MAX_PAGE_CHARS = 8000
MAX_TOTAL_CHARS = 20000

# Maximum amount of raw HTTP data we are willing to download
# from a single page.
MAX_RESPONSE_BYTES = 2_000_000


# ------------------------------------------------------------
# URL safety
# ------------------------------------------------------------

def is_safe_url(url: str) -> bool:
    """
    Check whether a URL is safe for server-side fetching.

    Only HTTP/HTTPS URLs are allowed.

    Hostnames resolving to local/private/reserved addresses
    are rejected to prevent SSRF against the local machine
    or private networks.
    """

    try:
        parsed = urlparse(url)

        # Only normal web protocols.
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname

        if not hostname:
            return False

        # Resolve the hostname ourselves so we can inspect
        # the actual destination IP addresses.
        addresses = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )

        if not addresses:
            return False

        for address in addresses:
            ip = ipaddress.ip_address(
                address[4][0]
            )

            # Reject anything that is not a normal public address.
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False

        return True

    except (
        ValueError,
        socket.gaierror,
        OSError,
    ):
        return False


# ------------------------------------------------------------
# Search
# ------------------------------------------------------------

def search_web(
    query: str,
    max_results: int = MAX_RESULTS,
):
    """
    Search DuckDuckGo and return raw search results.
    """

    results = []

    # Never allow callers to silently override our upper bound.
    max_results = min(
        max_results,
        MAX_RESULTS,
    )

    with DDGS() as ddgs:
        for result in ddgs.text(
            query,
            max_results=max_results,
        ):
            results.append(result)

    return results


# ------------------------------------------------------------
# Page fetching
# ------------------------------------------------------------

def fetch_page_text(
    url: str,
    max_chars: int = MAX_PAGE_CHARS,
):
    """
    Fetch and clean webpage text.

    Only public HTTP/HTTPS destinations are allowed.
    """

    # --------------------------------------------------------
    # Validate URL before making a request.
    # --------------------------------------------------------

    if not is_safe_url(url):
        print(
            f"[WEB] Blocked unsafe URL: {url}"
        )
        return ""

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers=HEADERS,
            allow_redirects=False,
        )

        response.raise_for_status()

        # ----------------------------------------------------
        # Prevent unexpectedly huge responses.
        # ----------------------------------------------------

        content_length = response.headers.get(
            "Content-Length"
        )

        if content_length:
            try:
                if int(content_length) > MAX_RESPONSE_BYTES:
                    print(
                        f"[WEB] Response too large: {url}"
                    )
                    return ""

            except ValueError:
                pass

        if len(response.content) > MAX_RESPONSE_BYTES:
            print(
                f"[WEB] Response exceeded byte limit: {url}"
            )
            return ""

        # ----------------------------------------------------
        # Parse HTML.
        # ----------------------------------------------------

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

    except requests.RequestException as exc:
        print(
            f"[WEB] Request failed: {url} "
            f"({type(exc).__name__})"
        )
        return ""

    except Exception as exc:
        print(
            f"[WEB] Page processing failed: {url} "
            f"({type(exc).__name__}: {exc})"
        )
        return ""


# ------------------------------------------------------------
# Connectivity
# ------------------------------------------------------------

def is_online():
    """
    Cheap connectivity check.
    """

    try:
        socket.create_connection(
            ("8.8.8.8", 53),
            timeout=CONNECTIVITY_TIMEOUT,
        )
        return True

    except OSError:
        return False


# ------------------------------------------------------------
# Full web lookup
# ------------------------------------------------------------

def web_lookup(
    query: str,
    max_results: int = MAX_RESULTS,
    page_chars: int = MAX_PAGE_CHARS,
    total_chars: int = MAX_TOTAL_CHARS,
) -> str:
    """
    Collect raw web evidence.

    This function does NOT summarize the results.
    The Observer performs summarization later.
    """

    if not is_online():
        print(
            "[WEB] Offline; skipping search."
        )
        return ""

    # Enforce global limits regardless of caller input.
    max_results = min(
        max_results,
        MAX_RESULTS,
    )

    page_chars = min(
        page_chars,
        MAX_PAGE_CHARS,
    )

    total_chars = min(
        total_chars,
        MAX_TOTAL_CHARS,
    )

    print(
        f"[WEB] Searching for: {query}"
    )

    context_chunks = []
    total_chars_used = 0

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

            # ------------------------------------------------
            # Validate before fetching.
            # ------------------------------------------------

            if not is_safe_url(url):
                print(
                    f"[WEB] Skipping unsafe URL: {url}"
                )
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

            chunk = (
                f"Source: {url}\n"
                f"{text}"
            )

            remaining = (
                total_chars
                - total_chars_used
            )

            if remaining <= 0:
                break

            # Don't allow the final chunk to push the
            # global evidence buffer over its limit.
            if len(chunk) > remaining:
                chunk = chunk[:remaining]

            context_chunks.append(chunk)

            total_chars_used += len(chunk)

            if total_chars_used >= total_chars:
                print(
                    f"[WEB] Global evidence limit reached: "
                    f"{total_chars_used}/{total_chars} chars"
                )
                break

    except Exception as exc:
        print(
            f"[WEB] Search failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return ""

    return "\n\n".join(
        context_chunks
    )