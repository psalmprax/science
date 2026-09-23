FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY apedp/ ./apedp/
COPY Makefile ./

# Install the package and its resolved dependencies.
RUN pip install --no-cache-dir .

EXPOSE 8000

# The API entry point is a factory function, so uvicorn needs `--factory`.
CMD ["uvicorn", "apedp.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
