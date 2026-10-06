# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash git grep findutils coreutils sed diffutils patch procps passwd \
    build-essential cmake ninja-build pkg-config gdb binutils \
    curl ca-certificates xz-utils \
    libfontconfig1 libfreetype6 libgraphite2-3 libharfbuzz0b \
    libicu72 libpng16-16 libssl3 zlib1g \
    fontconfig fonts-dejavu-core poppler-utils \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

ARG NODE_VERSION=24.21.0
ARG TYPESCRIPT_VERSION=5.9.3
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) NODE_ARCH='x64' ;; \
        arm64) NODE_ARCH='arm64' ;; \
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 3 \
        "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" \
        -o /tmp/node.tar.xz; \
    curl -fsSL --retry 3 \
        "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt" \
        -o /tmp/node-SHASUMS256.txt; \
    node_sha="$(awk -v name="node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" '$2 == name {print $1}' /tmp/node-SHASUMS256.txt)"; \
    test -n "$node_sha"; \
    printf '%s  %s\n' "$node_sha" /tmp/node.tar.xz | sha256sum -c -; \
    tar -xJf /tmp/node.tar.xz -C /usr/local \
        --strip-components=1 --no-same-owner; \
    rm /tmp/node.tar.xz /tmp/node-SHASUMS256.txt; \
    node --version; \
    npm --version

RUN npm install -g "typescript@${TYPESCRIPT_VERSION}" \
    && tsc --version \
    && npm cache clean --force

RUN python -m pip install --no-cache-dir \
    pytest numpy pandas scipy matplotlib

# Full Debian TeX Live: packages and fonts are installed at build time.
# This is a large layer (several GB), intentionally not a warmed partial cache.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        texlive-full latexmk \
    && mktexlsr \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 sandbox \
    && useradd --uid 1000 --gid sandbox --create-home --shell /bin/bash sandbox \
    && mkdir -p /workspace /opt/latex-tests \
    && chown sandbox:sandbox /workspace

# Compatibility helper: retains the previous --outdir interface.
# XeLaTeX is the default; latexmk handles repeat passes and bibliography tools.
COPY <<'PYTHON' /usr/local/bin/latex-compile
#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Compile LaTeX offline using TeX Live and latexmk.")
parser.add_argument("--outdir", default=".", help="Output directory; created automatically")
parser.add_argument("--engine", choices=("xelatex", "pdflatex", "lualatex"), default="xelatex")
parser.add_argument("--keep-logs", action="store_true", help="Accepted for compatibility; logs are always kept")
parser.add_argument("source", help="Path to one .tex file")
args = parser.parse_args()
source = Path(args.source).resolve()
if not source.is_file() or source.suffix.lower() != ".tex":
    parser.error(f"Expected an existing .tex file: {source}")
outdir = Path(args.outdir).resolve()
outdir.mkdir(parents=True, exist_ok=True)
for key, default in (
    ("TEXMFVAR", "/tmp/texmf-var"),
    ("TEXMFCONFIG", "/tmp/texmf-config"),
    ("XDG_CACHE_HOME", "/tmp/cache"),
):
    os.environ.setdefault(key, default)
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
# Resolve relative includes/images against the source directory.
os.chdir(source.parent)
command = [
    "/usr/bin/latexmk", "-norc", f"-{args.engine}",
    "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
    "-no-shell-escape", f"-outdir={outdir}", str(source),
]
os.execv(command[0], command)
PYTHON

RUN chmod 0755 /usr/local/bin/latex-compile

# Tests are checks of the installation, not a mechanism for caching packages.
COPY <<'TEX' /opt/latex-tests/smoke.tex
\documentclass[11pt]{article}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\newfontfamily\cyrillicfont{DejaVu Serif}
\usepackage[margin=20mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array}
\usepackage{xcolor,enumitem,listings,tikz}
\usetikzlibrary{calc,fit,positioning,arrows.meta,shapes.geometric,er,shapes.multipart}
\usepackage{hyperref}
\begin{document}
\section{Offline TeX Live test}\label{sec:test}
\textbf{Bold}, \textit{italic}, \textsl{slanted}, \textsf{sans}, \texttt{monospace}.
{\small Small \textit{italic}.}
{\scriptsize Script size \textit{italic}.}
{\cyrillicfont English and Cyrillic: Hello, Привет.}
\[\sum_{i=1}^{n} i = \frac{n(n+1)}{2},\qquad x\in\mathbb{R}.\]
\begin{tabular}{lr}
\toprule Item & Value \\
\midrule Test & 42 \\
\bottomrule
\end{tabular}
\begin{enumerate}[label=\alph*)]
\item First item.
\end{enumerate}
\begin{lstlisting}[language=Python,basicstyle=\ttfamily]
print("Hello")
\end{lstlisting}
\begin{tikzpicture}
\node[draw,rectangle split,rectangle split parts=2] (a)
  {Article\nodepart{second}\underline{article\_id}};
\node[draw,diamond,right=25mm of a] (r) {writes};
\node[draw,right=25mm of r] (b) {Author};
\draw (a) -- node[above] {$(1,n)$} (r);
\draw (r) -- node[above] {$(0,n)$} (b);
\node[draw,dashed,fit=(a)(r)(b),inner sep=8mm] {};
\coordinate (mid) at ($(a)!0.5!(b)$);
\draw[-{Stealth}] ($(mid)+(0,-15mm)$) -- ($(mid)+(0,-8mm)$);
\end{tikzpicture}
See Section~\ref{sec:test}.
\end{document}
TEX

COPY <<'TEX' /opt/latex-tests/standalone.tex
\documentclass[tikz,border=5pt]{standalone}
\usetikzlibrary{calc,fit,positioning,arrows.meta,shapes.geometric,shapes.multipart}
\begin{document}
\begin{tikzpicture}
\node[draw,rectangle split,rectangle split parts=2] (a)
  {Entity\nodepart{second}\small\textit{Attribute}};
\node[draw,ellipse,right=20mm of a] (b) {Other entity};
\draw[-{Stealth}] (a) -- node[above] {$(0,n)$} (b);
\node[draw,dashed,fit=(a)(b),inner sep=5mm] {};
\fill ($(a)!0.5!(b)$) circle[radius=1pt];
\end{tikzpicture}
\end{document}
TEX

ENV HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    TEXMFVAR=/tmp/texmf-var \
    TEXMFCONFIG=/tmp/texmf-config \
    XDG_CACHE_HOME=/tmp/cache

WORKDIR /workspace
USER sandbox

# Every compile below must work without network access as the runtime user.
RUN --network=none set -eux; \
    latex-compile --outdir /tmp/latex-test/article /opt/latex-tests/smoke.tex; \
    latex-compile --outdir /tmp/latex-test/diagram /opt/latex-tests/standalone.tex; \
    latex-compile --engine pdflatex --outdir /tmp/latex-test/pdftex /opt/latex-tests/standalone.tex; \
    latex-compile --engine lualatex --outdir /tmp/latex-test/luatex /opt/latex-tests/smoke.tex; \
    test -s /tmp/latex-test/article/smoke.pdf; \
    test -s /tmp/latex-test/diagram/standalone.pdf; \
    test -s /tmp/latex-test/pdftex/standalone.pdf; \
    test -s /tmp/latex-test/luatex/smoke.pdf; \
    pdfinfo /tmp/latex-test/article/smoke.pdf; \
    rm -rf /tmp/latex-test /tmp/texmf-var /tmp/texmf-config /tmp/cache

CMD ["bash"]