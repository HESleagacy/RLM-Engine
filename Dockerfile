# Linux containers support both amd64 and arm64 hosts without changing the engine.
FROM python:3.12-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/rlm

WORKDIR /app

RUN groupadd --gid 10001 rlm \
    && useradd --uid 10001 --gid rlm --create-home rlm

COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY demo/ ./demo/

RUN python -m pip install ".[demo]" "gunicorn==23.0.0"

FROM base AS test

RUN python -m pip install ".[dev]"
COPY tests/ ./tests/

USER rlm
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]

FROM base AS runtime

USER rlm
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3).close()"

# Synchronous workers avoid adding request threads around fork-based execution.
# Allow long reasoning runs; this is a web-worker timeout, not an engine budget.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--threads", "1", "--timeout", "6600", "--access-logfile", "-", "--error-logfile", "-", "demo.app:app"]
