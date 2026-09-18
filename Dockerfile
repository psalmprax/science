FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY README.md ./

# Install the package and its resolved dependencies. The previous build shelled out
# to `pip-compile ... || true`, which both hid a failing resolve and produced a
# requirements.txt that nothing consumed (`pip install .` resolves from pyproject).
RUN pip install --no-cache-dir .

COPY apedp/ ./apedp/
COPY Makefile ./

EXPOSE 8000

# The API entry point is a factory function, so uvicorn needs `--factory`.
# Without it uvicorn imports `apedp.api:create_app` and tries to serve the
# function object itself, and the container exits on boot.
CMD ["uvicorn", "apedp.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
