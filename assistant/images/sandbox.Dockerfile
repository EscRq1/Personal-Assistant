FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    grep \
    findutils \
    coreutils \
    sed \
    diffutils \
    patch \
    procps \
    passwd \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    gdb \
    binutils \
    curl \
    ca-certificates \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

ARG NODE_VERSION=24.21.0
ARG TYPESCRIPT_VERSION=5.9.3

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
        amd64) NODE_ARCH='x64' ;; \
        arm64) NODE_ARCH='arm64' ;; \
        *) echo "Unsupported architecture: $arch"; exit 1 ;; \
    esac; \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz"; \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt"; \
    grep " node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz\$" SHASUMS256.txt | sha256sum -c -; \
    tar -xJf "node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" \
        -C /usr/local \
        --strip-components=1 \
        --no-same-owner; \
    rm "node-v${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz" SHASUMS256.txt; \
    node --version; \
    npm --version

RUN npm install -g "typescript@${TYPESCRIPT_VERSION}" \
    && tsc --version

RUN python -m pip install --no-cache-dir \
    pytest \
    numpy \
    pandas \
    scipy \
    matplotlib

RUN groupadd --gid 1000 sandbox \
    && useradd \
        --uid 1000 \
        --gid sandbox \
        --create-home \
        --shell /bin/bash \
        sandbox \
    && mkdir /workspace \
    && chown sandbox:sandbox /workspace

ENV HOME=/tmp \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace
USER sandbox

CMD ["bash"]