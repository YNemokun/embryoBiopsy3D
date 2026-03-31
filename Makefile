SHELL := /bin/bash
PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install install-runtime test clean

# Editable install with test extras (matches README)
install:
	$(PIP) install -e ".[test]"

# Library only, no pytest extras
install-runtime:
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest tests/

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + || true
	find . -name ".pytest_cache" -type d -exec rm -rf {} + || true
	find . -name ".ruff_cache" -type d -exec rm -rf {} + || true
	rm -rf build/ dist/ *.egg-info
