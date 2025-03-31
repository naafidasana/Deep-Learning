# Default Variables
VENV_NAME := dl_uv
CONDA_ENV_NAME := dl_conda
VENV_PYTHON := $(VENV_NAME)/bin/python
PYTHON := 3.12


.PHONY: setup conda_setup clean help

# UV Environment Setup
setup:
	@echo "Setting up development environment..."
	@if ! command -v uv > /dev/null; then \
		echo "Error: UV package manager not found. Please install it first."; \
		echo "curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@echo "Creating virtual environment with UV..."
	@uv venv $(VENV_NAME)
	@echo "Installing project dependencies from pyproject.toml..."
	@uv pip install --python $(VENV_PYTHON) -r requirements.txt
	@echo "Setup completed successfully! Activate with: source ./$(VENV_NAME)/bin/activate"

# Conda Environment Setup
conda_setup:
	@echo "Creating Conda environment '$(CONDA_ENV_NAME)'..."
	@conda create -n $(CONDA_ENV_NAME) python=$(PYTHON) -y || (echo "Failed. Ensure Conda is installed." && exit 1)
	@conda activate $(CONDA_ENV_NAME)
	@echo "Installing project dependencies from pyproject.toml..."
	@pip install -r requirements.txt
	@echo "✅ Done. Activate with: conda activate $(CONDA_ENV_NAME)"

# Cleanup
clean: clean_venv clean_conda clean_pyc
	@echo "🧹 All environments and cache cleaned"

clean_venv:
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "Removing UV environment '$(VENV_NAME)'..."; \
		rm -rf $(VENV_NAME); \
	fi

clean_conda:
	@if conda env list | grep -q "$(CONDA_ENV_NAME)"; then \
		echo "Removing Conda environment '$(CONDA_ENV_NAME)'..."; \
		conda remove -n $(CONDA_ENV_NAME) --all -y; \
	fi

clean_pyc:
	@find . -name "*.pyc" -delete -o -name "__pycache__" -exec rm -rf {} +

# Help
help:
	@echo "Deep Learning Project - Makefile"
	@echo "--------------------------------"
	@echo "make setup        # Create UV env 'dl_uv' (Python 3.12)"
	@echo "make conda_setup  # Create Conda env 'dl_conda' (Python 3.12)"
	@echo "make clean        # Remove all environments and cache"
	@echo "make clean_venv   # Remove only UV environment"
	@echo "make clean_conda  # Remove only Conda environment"
	@echo "make clean_pyc    # Remove Python cache files"