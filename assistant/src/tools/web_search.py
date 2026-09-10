from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
import socket
import ipaddress
from urllib.parse import urlparse


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://duckduckgo.com/",
}

# ------------------------------------------------------------
# Safety / resource limits
# ------------------------------------------------------------

REQUEST_TIMEOUT = 10
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
    are rejected to reduce SSRF risk.
    """

    try:
        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname

        if not hostname:
            return False

        addresses = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )

        if not addresses:
            return False

        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])

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

    max_results = max(
        1,
        min(max_results, MAX_RESULTS),
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

    Redirects are followed, but the final destination is
    validated before its content is accepted.
    """

    # --------------------------------------------------------
    # Validate initial URL before making a request.
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
            allow_redirects=True,
        )

        # ----------------------------------------------------
        # Validate the final redirect destination.
        # ----------------------------------------------------

        final_url = response.url

        if final_url != url:
            print(
                f"[WEB] Redirected: {url} -> {final_url}"
            )

        if not is_safe_url(final_url):
            print(
                f"[WEB] Blocked unsafe redirect: {final_url}"
            )
            return ""

        # ----------------------------------------------------
        # Raise for HTTP errors.
        # ----------------------------------------------------

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
                        f"[WEB] Response too large: "
                        f"{final_url} "
                        f"({content_length} bytes)"
                    )
                    return ""

            except ValueError:
                pass

        if len(response.content) > MAX_RESPONSE_BYTES:
            print(
                f"[WEB] Response exceeded byte limit: "
                f"{final_url} "
                f"({len(response.content)} bytes)"
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
            [
                "script",
                "style",
                "nav",
                "footer",
                "header",
                "noscript",
            ]
        ):
            tag.decompose()

        text = " ".join(
            soup.stripped_strings
        )

        return text[:max_chars]

    except requests.HTTPError as exc:
        status = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )

        reason = (
            exc.response.reason
            if exc.response is not None
            else ""
        )

        print(
            f"[WEB] HTTP {status}"
            f"{f' ({reason})' if reason else ''}: "
            f"{url}"
        )
        return ""

    except requests.Timeout:
        print(
            f"[WEB] Request timed out: {url}"
        )
        return ""

    except requests.ConnectionError as exc:
        print(
            f"[WEB] Connection failed: "
            f"{url} ({type(exc).__name__})"
        )
        return ""

    except requests.RequestException as exc:
        print(
            f"[WEB] Request failed: "
            f"{url} ({type(exc).__name__}: {exc})"
        )
        return ""

    except Exception as exc:
        print(
            f"[WEB] Page processing failed: "
            f"{url} "
            f"({type(exc).__name__}: {exc})"
        )
        return ""


# ------------------------------------------------------------
# Connectivity
# ------------------------------------------------------------

def is_online():
    """
    Cheap connectivity check.

    This is only a quick network sanity check; the actual
    web request remains authoritative.
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

    # --------------------------------------------------------
    # Enforce global limits regardless of caller input.
    # --------------------------------------------------------

    max_results = max(
        1,
        min(max_results, MAX_RESULTS),
    )

    page_chars = max(
        1,
        min(page_chars, MAX_PAGE_CHARS),
    )

    total_chars = max(
        1,
        min(total_chars, MAX_TOTAL_CHARS),
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

        if not search_results:
            print(
                "[WEB] Search returned no results."
            )
            return ""

        print(
            f"[WEB] Search returned "
            f"{len(search_results)} result(s)"
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

            # ------------------------------------------------
            # Keep final evidence buffer within its limit.
            # ------------------------------------------------

            if len(chunk) > remaining:
                chunk = chunk[:remaining]

            context_chunks.append(chunk)

            total_chars_used += len(chunk)

            if total_chars_used >= total_chars:
                print(
                    "[WEB] Global evidence limit reached: "
                    f"{total_chars_used}/{total_chars} chars"
                )
                break

    except Exception as exc:
        print(
            "[WEB] Search failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return ""

    combined = "\n\n".join(
        context_chunks
    )

    print(
        f"[WEB] Combined evidence: "
        f"{len(combined)} chars"
    )

    return combined