.PHONY: install test lint format clean clean-cache

install:
	pip install -e .

test:
	pytest tests/ -v

lint:
	ruff check apedp/ tests/

format:
	ruff format apedp/ tests/

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .ruff_cache/ .mypy_cache/

clean-cache:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
