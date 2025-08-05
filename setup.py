"""
Setup script for Technical Document AI
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name="technical-doc-ai",
    version="1.0.0",
    description="AI system for processing technical documents and answering complex questions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Technical Doc AI Team",
    author_email="info@technicaldocai.com",
    url="https://github.com/yourusername/technical-doc-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-mock>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "technical-doc-ai=main:main",
            "run-ui=run_app:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Markup",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai, nlp, pdf, documents, technical, building-codes, calculations",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/technical-doc-ai/issues",
        "Source": "https://github.com/yourusername/technical-doc-ai",
        "Documentation": "https://technical-doc-ai.readthedocs.io",
    },
    include_package_data=True,
    zip_safe=False,
)