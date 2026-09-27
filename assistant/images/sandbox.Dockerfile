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
    && rm -rf /var/lib/apt/lists/*

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