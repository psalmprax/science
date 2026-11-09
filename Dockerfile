FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain 1.99.0
ENV PATH="/root/.cargo/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY rust/ ./rust/
RUN pip install --no-cache-dir "maturin>=1.7,<2.0" && \
    maturin build --release --locked --manifest-path rust/apedp-kernel/Cargo.toml --out /build/wheels

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md Makefile ./
COPY apedp/ ./apedp/
COPY apedp_dyn/ ./apedp_dyn/
COPY specifications/ ./specifications/
COPY --from=builder /build/wheels/*.whl /tmp/wheels/

# Install the package, native kernel wheel, and resolved dependencies, then drop root privileges.
RUN pip install --no-cache-dir . /tmp/wheels/*.whl && rm -rf /tmp/wheels && \
    useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

# The API entry point is a factory function, so uvicorn needs `--factory`.
CMD ["uvicorn", "apedp.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
