.PHONY: help venv setup-cuda install install-cuda install-cpu test test-single train minimal comparison fevbench gifteval notebook lint format clean

PYTHON ?= python3
UV ?= uv
VENV := .venv
VENV_PYTHON := $(VENV)/bin/python
CUDA_ENV_FILE := $(HOME)/.cache/tirex2/cuda/env.sh

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## Create the project virtual environment with uv
	$(UV) venv $(VENV)

$(CUDA_ENV_FILE): ## Detect hardware and download a local CUDA 12.8 toolkit if needed
	@echo "Setting up CUDA 12.8 toolchain for the detected GPU/driver..."
	$(PYTHON) scripts/setup_hardware.py --download --env-file $(CUDA_ENV_FILE)

setup-cuda: $(CUDA_ENV_FILE) ## Alias for creating $(CUDA_ENV_FILE)

install: ## Auto-detect GPU and install project (CUDA if available, otherwise CPU)
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "NVIDIA GPU detected; installing CUDA variant."; \
		$(MAKE) install-cuda; \
	else \
		echo "No NVIDIA GPU detected; installing CPU variant."; \
		$(MAKE) install-cpu; \
	fi

install-cuda: $(CUDA_ENV_FILE) ## Install editable project with CUDA 12.8 / torch cu128
	$(UV) venv $(VENV)
	$(UV) pip install -r requirements/torch-cu128.txt
	$(UV) pip install -r requirements/build.txt
	$(UV) pip install setuptools wheel ruff pyupgrade build pytest pytest-asyncio
	. $(CUDA_ENV_FILE) && $(UV) pip install -e ".[examples,gluonts,fev]"

install-cpu: ## Install editable project with CPU-only torch
	$(UV) venv $(VENV)
	$(UV) pip install -r requirements/torch-cpu.txt
	$(UV) pip install setuptools wheel ruff pyupgrade build pytest pytest-asyncio
	$(UV) pip install -e ".[examples,gluonts,fev]"

test: ## Run the full pytest suite
	$(UV) run pytest test/

test-single: ## Run a single test file: make test-single FILE=test/test_forecast_model.py
	$(UV) run pytest $(FILE)

train: ## Run the training CLI: make train ARGS="--config configs/finetune.yaml --out ./checkpoints"
	$(VENV_PYTHON) scripts/train.py $(ARGS)

minimal: ## Run the sine-wave smoke example
	$(VENV_PYTHON) examples/sine_wave.py

comparison: ## Run the covariate forecast comparison demo
	$(VENV_PYTHON) examples/covariate_forecasts.py

fevbench: ## Run fev-bench: make fevbench ARGS="--tasks examples/fevbench/tasks-mini.yaml --out output/fevbench-mini.csv"
	PYTHONPATH=examples/gifteval:$(PYTHONPATH) $(VENV_PYTHON) examples/fevbench/run_fevbench.py $(ARGS)

gifteval: ## Run GiftEval: make gifteval ARGS="/path/to/gifteval_storage pretrained --out output/gifteval.csv"
	PYTHONPATH=examples/gifteval:$(PYTHONPATH) $(VENV_PYTHON) examples/gifteval/run_gifteval.py $(ARGS)

notebook: ## Launch Jupyter Lab with the getting started notebook
	$(VENV_PYTHON) -m jupyter lab examples/getting_started.ipynb

lint: ## Run ruff checks
	$(VENV_PYTHON) -m ruff check src test

format: ## Run ruff formatter
	$(VENV_PYTHON) -m ruff format src test

clean: ## Remove venv, outputs, and compiled artifacts
	rm -rf $(VENV) output .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
