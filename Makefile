# Makefile for Credit Risk Scoring Project
# Common development and deployment tasks

.PHONY: help install test lint format train api docker-build docker-run clean

# Default target
help:
	@echo "Credit Risk Scoring - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install      - Install dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting (black, isort, flake8, mypy)"
	@echo "  format       - Format code with black and isort"
	@echo ""
	@echo "Data & Training:"
	@echo "  download     - Download German Credit dataset"
	@echo "  train        - Run full training pipeline"
	@echo "  infer        - Run inference on new data"
	@echo ""
	@echo "API:"
	@echo "  api          - Run API server locally"
	@echo "  api-dev      - Run API with auto-reload"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docker-compose-up - Start all services"
	@echo "  docker-compose-down - Stop all services"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        - Clean build artifacts"
	@echo "  check-env    - Check environment setup"

# Install dependencies
install:
	python3 -m pip install --upgrade pip
	python3 -m pip install -r requirements.txt
	pre-commit install

# Run tests
test:
	PYTHONPATH="$(CURDIR)/src" pytest tests/ -v

test-cov:
	PYTHONPATH="$(CURDIR)/src" pytest tests/ -v --cov=credit_risk --cov-report=html --cov-report=term-missing

# Linting
lint:
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Download dataset
download:
	mkdir -p data/raw
	curl -s "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data" \
		-o data/raw/german_credit.data
	@echo "Dataset downloaded to data/raw/german_credit.data"

# Training pipeline
train: download
	PYTHONPATH="$(CURDIR)/src" python3 -m credit_risk.pipeline --output-dir outputs/$$(date +%Y%m%d_%H%M%S)

# Inference
infer:
	@read -p "Input file path: " input; \
	read -p "Model directory: " model_dir; \
	read -p "Output file path: " output; \
	PYTHONPATH="$(CURDIR)/src" python3 -m credit_risk.pipeline --inference --input $$input --model-dir $$model_dir --output $$output

# API server
api:
	PYTHONPATH="$(CURDIR)/src" python3 -m credit_risk.api

api-dev:
	PYTHONPATH="$(CURDIR)/src" uvicorn credit_risk.api:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-build:
	docker build -t credit-risk-api:latest .

docker-run:
	docker run -d \
		--name credit-risk-api \
		-p 8000:8000 \
		-v $(PWD)/models:/app/models:ro \
		-v $(PWD)/data:/app/data:ro \
		-v $(PWD)/logs:/app/logs \
		credit-risk-api:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

# Clean
clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	rm -rf src/credit_risk/__pycache__ src/credit_risk/**/__pycache__
	rm -rf tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Check environment
check-env:
	@echo "Python version:"
	@python3 --version
	@echo ""
	@echo "Key packages:"
	@python3 -c "import numpy; print(f'numpy: {numpy.__version__}')"
	@python3 -c "import pandas; print(f'pandas: {pandas.__version__}')"
	@python3 -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
	@python3 -c "import fastapi; print(f'fastapi: {fastapi.__version__}')"
	@echo ""
	@echo "Project structure:"
	@ls -la src/credit_risk/

# Run full pipeline with logging
pipeline:
	python3 -m credit_risk.pipeline --log-level DEBUG 2>&1 | tee logs/pipeline_$(shell date +%Y%m%d_%H%M%S).log

# Generate requirements.txt from current environment
freeze:
	pip freeze > requirements-frozen.txt

# Security scan
security:
	python3 -m pip install safety bandit
	safety check
	bandit -r src/

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Jupyter notebook
notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root