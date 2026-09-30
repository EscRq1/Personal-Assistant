import ipaddress
import re
import secrets
import socket
import time
from urllib import robotparser
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

try:
    import trafilatura
except ImportError:
    trafilatura = None

# Headers / limits -------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
              "image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Optional Jina Reader API key (free tier: 200 req/min, 10M tokens);
# keyless: 20 req/min.
JINA_API_KEY = ""

REQUEST_TIMEOUT = 10
FETCH_DEADLINE = 30.0
MAX_RESULTS = 5
MAX_PAGE_CHARS = 8000
MAX_TOTAL_CHARS = 20000
MAX_RESPONSE_BYTES = 2_000_000
MAX_REDIRECTS = 4
MAX_FETCH_CHARS = 24_000
MAX_JINA_CHARS = 300_000
MAX_URL_LEN = 2048
JINA_TIMEOUT = 25
DOMAIN_RATE_SECONDS = 5.0
ROBOTS_CACHE_TTL = 3600
ROBOTS_MAX_BYTES = 200_000
CACHE_MAX_ENTRIES = 1024

# Hosts we refuse to fetch (prevents forwarding internal URLs through
# third-party reader proxies).
BLOCKED_PROXY_HOSTS = ("r.jina.ai",)

# One session for every outbound request. trust_env=False = no .netrc
# credentials, no ambient HTTP(S)_PROXY / NO_PROXY, no environment
# auth — the fetcher is hermetic by construction.
_session = requests.Session()
_session.trust_env = False


# URL safety -------------------------------------------------------

def _is_public_ip(ip_str: str) -> bool:
    """True only for globally-routable addresses. Unwraps IPv4-mapped
    IPv6 (e.g. ::ffff:127.0.0.1) so the is_global check sees the
    embedded IPv4 address."""
    ip = ipaddress.ip_address(ip_str)
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return ip.is_global


def assert_public_url(url: str) -> str:
    """Raise ValueError unless url is a plain http(s) URL on port
    80/443 without credentials whose every resolved address is
    globally routable. Rejects before any DNS or network activity:
    non-http(s) schemes, missing hostnames, backslashes/control
    characters, URL credentials (user:pass@host), reader-proxy hosts,
    explicit ports other than 80/443 (including :0), over-long URLs."""
    if not isinstance(url, str) or not url:
        raise ValueError("URL must be a non-empty string")
    if len(url) > MAX_URL_LEN:
        raise ValueError("URL too long")
    if "\\" in url or any(ord(c) < 0x20 or c == "\x7f" for c in url):
        raise ValueError("URL contains disallowed characters")
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError(f"URL not allowed: {url!r}")
    if p.username or p.password:
        raise ValueError("URL credentials not allowed")
    h = p.hostname.lower()
    if any(h == b or h.endswith("." + b) for b in BLOCKED_PROXY_HOSTS):
        raise ValueError("fetching through reader proxies is not allowed")
    if p.port is not None and p.port not in (80, 443):
        raise ValueError(f"port {p.port} not allowed")
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(p.hostname, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError) as e:
        raise ValueError(f"DNS resolution failed for {p.hostname}: {e}")
    if not infos:
        raise ValueError(f"DNS resolution failed for {p.hostname}")
    for info in infos:
        if not _is_public_ip(info[4][0]):
            raise ValueError(f"{p.hostname} resolves to non-public IP")
    return url


# Fetching ---------------------------------------------------------

def fetch_page(
    url: str,
    max_bytes: int = MAX_RESPONSE_BYTES,
    timeout: float = REQUEST_TIMEOUT,
    max_redirects: int = MAX_REDIRECTS,
    deadline: float = FETCH_DEADLINE,
) -> tuple[bytes, str]:
    """GET a URL safely. Returns (body_bytes, final_url). The URL is
    validated before EVERY hop; the body is streamed and aborted the
    moment max_bytes is exceeded. The wall-clock deadline is checked
    before each hop and at every chunk boundary — it is NOT a hard
    kill (see KNOWN LIMITS). Raises ValueError for policy
    violations, requests exceptions for transport failures."""
    start = time.monotonic()
    cur = url
    for _ in range(max_redirects + 1):
        if time.monotonic() - start > deadline:
            raise ValueError(f"deadline exceeded: {url}")
        assert_public_url(cur)
        with _session.get(
            cur, headers=HEADERS, timeout=timeout,
            allow_redirects=False, stream=True,
        ) as r:
            if r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("Location")
                if not loc:
                    raise ValueError("redirect without Location header")
                cur = urljoin(cur, loc)  # handles relative Location
                continue
            r.raise_for_status()
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError(f"response too large: {url}")
                if time.monotonic() - start > deadline:
                    raise ValueError(f"deadline exceeded: {url}")
                chunks.append(chunk)
        return b"".join(chunks), cur
    raise ValueError(f"too many redirects from {url}")


# Politeness: per-domain rate limit + robots.txt
# (direct fetches only; Jina does its own)

_last_domain_fetch: dict[str, float] = {}
_robots_cache: dict[str, tuple[float, list[str] | None]] = {}


def _domain_rate_limited(url: str) -> bool:
    """True if this domain was fetched less than DOMAIN_RATE_SECONDS
    ago; records the timestamp when allowed. PAGE fetches only —
    robots.txt is paced by its own per-origin cache, not this
    limiter."""
    if len(_last_domain_fetch) >= CACHE_MAX_ENTRIES:
        _last_domain_fetch.clear()  # crude eviction; single-user box
    host = urlparse(url).hostname or ""
    now = time.monotonic()
    last = _last_domain_fetch.get(host)
    if last is not None and now - last < DOMAIN_RATE_SECONDS:
        return True
    _last_domain_fetch[host] = now
    return False


def _robots_allowed(url: str) -> bool:
    """Check robots.txt for url (cached per origin, 1h TTL). The
    robots fetch does NOT touch the page rate limiter — the per-origin
    cache itself bounds robots traffic to one fetch per hour. A failed
    or non-2xx robots fetch soft-fails as 'allowed' for this call
    (RFC 9309) but is NOT cached, so a transient failure never becomes
    a cached permission."""
    p = urlparse(url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    now = time.monotonic()
    entry = _robots_cache.get(robots_url)
    if entry is None or now - entry[0] > ROBOTS_CACHE_TTL:
        try:
            data, _ = fetch_page(robots_url, max_bytes=ROBOTS_MAX_BYTES)
            lines = (data.decode("utf-8", "replace").splitlines()
                     or None)
        except requests.RequestException:
            return True  # 404/5xx or transport failure: soft-fail, no cache
        entry = (now, lines)
        if len(_robots_cache) >= CACHE_MAX_ENTRIES:
            _robots_cache.clear()
        _robots_cache[robots_url] = entry
    if entry[1] is None:
        return True
    rp = robotparser.RobotFileParser()
    rp.parse(entry[1])
    return rp.can_fetch(HEADERS["User-Agent"], url)


# Text extraction --------------------------------------------------

def extract_text(html, max_chars: int) -> str:
    """Generic bs4 strip — good enough for search-result snippets.
    Accepts bytes (charset auto-detected from Content-Type/meta) or
    str."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer",
                     "header", "noscript"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)[:max_chars]


def _extract_main_content(html) -> str:
    """Main-content extraction for the single-URL deep fetch.
    trafilatura when installed, bs4 fallback. Accepts bytes or str."""
    if trafilatura is not None:
        text = trafilatura.extract(html, include_comments=False,
                                   include_tables=True)
        if text:
            return text
    return extract_text(html, MAX_FETCH_CHARS)


# Search -----------------------------------------------------------

def search_web(query: str, max_results: int = MAX_RESULTS):
    """Search DuckDuckGo and return raw results."""
    max_results = max(1, min(max_results, MAX_RESULTS))
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _fetch_snippet(url: str, page_chars: int) -> str:
    """Best-effort page text for one search result. Returns "" when
    rate-limited / robots-blocked / unreachable / empty; the caller
    then falls back to the DDGS snippet."""
    if _domain_rate_limited(url):
        print(f"[WEB] Rate limited, skipping: {url}")
        return ""
    if not _robots_allowed(url):
        print(f"[WEB] Disallowed by robots.txt: {url}")
        return ""
    try:
        html, _ = fetch_page(url)
        text = extract_text(html, page_chars)
    except (ValueError, requests.RequestException) as e:
        print(f"[WEB] Fetch failed: {url} ({e.__class__.__name__}: {e})")
        return ""
    print(f"[WEB] Extracted: {len(text)} chars")
    return text


def web_lookup(
    query: str,
    max_results: int = MAX_RESULTS,
    page_chars: int = MAX_PAGE_CHARS,
    total_chars: int = MAX_TOTAL_CHARS,
) -> str:
    """Collect raw web evidence for the Observer to summarize. The
    returned string is untrusted data — wrap it with wrap_evidence()
    before it enters the model context. An empty return means "no
    data" (offline / blocked / no results), NOT "nothing
    relevant"; the Observer prompt should treat it as such."""
    max_results = max(1, min(max_results, MAX_RESULTS))
    page_chars = max(1, min(page_chars, MAX_PAGE_CHARS))
    total_chars = max(1, min(total_chars, MAX_TOTAL_CHARS))

    print(f"[WEB] Searching for: {query}")
    try:
        search_results = search_web(query, max_results=max_results)
    except Exception as e:
        print(f"[WEB] Search failed: {type(e).__name__}: {e}")
        return ""
    if not search_results:
        print("[WEB] Search returned no results.")
        return ""
    print(f"[WEB] Search returned {len(search_results)} result(s)")

    chunks, used = [], 0
    for result in search_results:
        url = result.get("href") or result.get("url")
        if not url:
            continue
        text = _fetch_snippet(url, page_chars)
        parts = [f"Source: {url}"]
        if result.get("title"):
            parts[0] += f"\nTitle: {result['title']}"
        if text:
            parts.append(text)
        elif result.get("body"):
            # Page fetch failed/empty — keep the DDGS snippet as a
            # cheap fallback signal.
            parts.append(f"Snippet: {result['body']}")
        else:
            continue
        chunk = "\n".join(parts)
        remaining = total_chars - used
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        used += len(chunk)
        if used >= total_chars:
            print(f"[WEB] Global evidence limit reached: "
                  f"{used}/{total_chars} chars")
            break

    combined = "\n\n".join(chunks)
    print(f"[WEB] Combined evidence: {len(combined)} chars")
    return combined


# Jina Reader (primary fetch path for web_fetch) -------------------

def _fetch_jina(url: str) -> tuple[str | None, int | None]:
    """Fetch url via Jina Reader. Returns (markdown | None, status).

    Jina fetches from their datacenter IPs (not yours), renders JS,
    and applies its own robots policy. The target URL is fully
    percent-encoded so query strings survive the hop; redirects are
    rejected outright (fixed endpoint). An empty body or transport
    failure returns (None, status) so the caller falls through to the
    direct path.
    """
    jina_url = "https://r.jina.ai/" + quote(url, safe="")
    headers = {**HEADERS, "x-respond-with": "markdown",
               "Accept": "text/markdown"}
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"
    deadline = time.monotonic() + JINA_TIMEOUT
    try:
        with _session.get(jina_url, headers=headers, timeout=JINA_TIMEOUT,
                          allow_redirects=False, stream=True) as r:
            status = r.status_code
            if status != 200:
                return None, status
            chunks, size = [], 0
            for chunk in r.iter_content(65536):
                size += len(chunk)
                if size > MAX_JINA_CHARS or time.monotonic() > deadline:
                    return None, status
                chunks.append(chunk)
            text = b"".join(chunks).decode("utf-8", "replace")
            if not text.strip():
                return None, status  # empty 200 — fall through to direct
            return text, status
    except requests.RequestException:
        return None, None


# Single-URL deep fetch (the web_fetch tool) -----------------------

def web_fetch(url: str) -> dict:
    """Fetch one known URL: Jina Reader primary (their IPs hit the
    site, not yours), direct guarded fetch as fallback.

    Returns a status dict, never a silent failure:
    {"ok", "url", "final_url", "provider" ("jina"|"direct"|None),
     "content", "truncated", "jina_status", "error"}. jina_status is
    surfaced even when the direct path succeeds, so callers/audit can
    tell "Jina was down (429/5xx)" from "Jina worked". final_url is
    the post-redirect destination on the direct path.

    BLOCKING — call via asyncio.to_thread() from async code.
    """
    result = {"ok": False, "url": url, "final_url": None,
              "provider": None, "content": "", "truncated": False,
              "jina_status": None, "error": None}

    try:
        assert_public_url(url)
    except ValueError as e:
        result["error"] = f"blocked: {e}"
        return result

    # --- Primary: Jina Reader ---
    jina_text, result["jina_status"] = _fetch_jina(url)
    if jina_text is not None:
        result.update(
            ok=True, provider="jina",
            content=jina_text[:MAX_FETCH_CHARS],
            truncated=len(jina_text) > MAX_FETCH_CHARS,
        )
        return result

    # --- Fallback: direct guarded fetch ---
    try:
        if _domain_rate_limited(url):
            result["error"] = "rate limited (try again in a few seconds)"
            return result
        if not _robots_allowed(url):
            result["error"] = "disallowed by robots.txt"
            return result
        html, final_url = fetch_page(url)
        result["final_url"] = final_url
        content = _extract_main_content(html)
    except ValueError as e:
        result["error"] = f"blocked: {e}"
        return result
    except requests.RequestException as e:
        result["error"] = f"fetch failed: {type(e).__name__}"
        return result

    if not content:
        result["error"] = "no extractable text"
        return result

    result.update(
        ok=True, provider="direct",
        content=content[:MAX_FETCH_CHARS],
        truncated=len(content) > MAX_FETCH_CHARS,
    )
    return result


# Untrusted-data wrappers (use when feeding results to the model) --

def _clean_meta(s) -> str:
    """Collapse whitespace/control chars in metadata fields (url,
    error, provider) so they cannot inject new lines into the wrapper
    framing."""
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _untrusted_block(body: str, footer: str) -> str:
    """Nonce-delimited block: fetched content cannot forge the END
    marker (the nonce is generated here, after the body exists)."""
    nonce = secrets.token_hex(8)
    return (f"<<<UNTRUSTED-{nonce}>>>\n{body}\n"
            f"<<<END-UNTRUSTED-{nonce}>>>\n{footer}")


def wrap_untrusted(result: dict) -> str:
    """Format a web_fetch result for model context. Metadata fields
    are whitespace-collapsed (a URL containing newlines cannot break
    framing); the failure path is emitted without a block. This is
    framing, not an injection control — see module docstring."""
    if not result.get("ok"):
        return (f"[web_fetch] failed for {_clean_meta(result.get('url'))}: "
                f"{_clean_meta(result.get('error')) or 'unknown error'}")
    final = result.get("final_url")
    src = _clean_meta(final) if final else _clean_meta(result.get("url"))
    return _untrusted_block(
        result["content"],
        f"(source: {src}, provider: {_clean_meta(result.get('provider'))}, "
        f"truncated: {result.get('truncated')}, "
        f"jina_status: {result.get('jina_status')})",
    )


def wrap_evidence(text: str,
                  source: str = "duckduckgo search results") -> str:
    """Wrap web_lookup's combined string before it enters the context.
    Empty input returns an explicit no-data marker so the model treats
    it as 'no evidence', not as silence to fill in."""
    if not text:
        return ("[web_lookup] no evidence collected "
                "(offline, blocked, or no results)")
    return _untrusted_block(text, f"(source: {source})")