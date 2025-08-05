# Technical Document AI - Makefile

.PHONY: help install install-dev test test-cov lint format clean run-ui setup-spacy docs

# Default target
help:
	@echo "Technical Document AI - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo "  make setup-spacy  - Download required spaCy models"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linting (flake8)"
	@echo "  make format       - Format code (black)"
	@echo "  make type-check   - Run type checking (mypy)"
	@echo ""
	@echo "Application:"
	@echo "  make run-ui       - Start Streamlit web interface"
	@echo "  make run-api      - Start FastAPI server (future)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docs         - Generate documentation"
	@echo ""

# Installation
install:
	@echo "📦 Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "📦 Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock black flake8 mypy pre-commit

setup-spacy:
	@echo "📚 Downloading spaCy models..."
	python -m spacy download en_core_web_sm

# Development
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "🔍 Running linting..."
	python -m flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	@echo "🎨 Formatting code..."
	python -m black src/ tests/ --line-length=88

type-check:
	@echo "🔍 Running type checking..."
	python -m mypy src/ --ignore-missing-imports

# Application
run-ui:
	@echo "🚀 Starting Streamlit web interface..."
	@echo "📍 URL: http://localhost:8501"
	python run_app.py

run-api:
	@echo "🚀 Starting FastAPI server..."
	@echo "📍 URL: http://localhost:8000"
	@echo "📖 Docs: http://localhost:8000/docs"
	# uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Utilities
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
	rm -rf chroma_db/
	rm -rf logs/

docs:
	@echo "📖 Generating documentation..."
	@echo "Documentation available in README.md"

# Quick setup for new environments
setup: install setup-spacy
	@echo "✅ Setup complete! Run 'make run-ui' to start the application."

# Development workflow
dev-setup: install-dev setup-spacy
	@echo "🛠️ Development environment setup complete!"
	@echo "✅ Run 'make test' to verify installation"
	@echo "🚀 Run 'make run-ui' to start the application"

# Full quality check
check: lint type-check test
	@echo "✅ All quality checks passed!"

# Pre-commit setup
pre-commit:
	@echo "🔧 Setting up pre-commit hooks..."
	pre-commit install