# Technical Document AI

An intelligent AI system for processing technical documents (building codes, standards, regulations) and answering complex questions requiring both semantic understanding and mathematical computation.

## Features

- **PDF Processing**: Robust extraction of text, tables, formulas, and document structure
- **Intelligent Search**: Semantic search using OpenAI embeddings and ChromaDB
- **Mathematical Reasoning**: Automatic calculations with SymPy and unit conversions with Pint
- **Natural Language Q&A**: GPT-4 powered query understanding and response generation
- **Source Tracking**: Full citation and reasoning transparency
- **Multi-format Support**: Tables, formulas, hierarchical sections, and cross-references

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd technical-doc-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd technical-doc-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Download spaCy model
make setup-spacy

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Web Interface

```bash
# Start the Streamlit web application
make run-ui
# or
python run_app.py

# Open http://localhost:8501 in your browser
```

### 3. Command Line Usage

```python
from src.main import TechnicalDocumentAI

# Initialize the system
ai = TechnicalDocumentAI()

# Ingest a document
doc_id = ai.ingest_document("path/to/building_code.pdf")

# Ask questions
answer = ai.answer_question("How much ventilation is required for a 6-car carpark?")
print(answer['answer'])
```

### 4. Development Setup

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Run with coverage
make test-cov

# Format code
make format

# Run linting
make lint

# Clean database only
python cleanup_db.py

# Comprehensive cleanup (all databases, caches, logs)
python cleanup_all.py

# Force cleanup without confirmation
python cleanup_all.py --force
```

## Architecture

### Document Processing Pipeline
- **PDF Parser**: Extracts text with layout preservation
- **Table Extractor**: Uses Camelot for structured table extraction
- **Formula Extractor**: Identifies mathematical expressions
- **Hierarchy Builder**: Constructs document structure tree

### Knowledge Base
- **Vector Store**: ChromaDB with OpenAI embeddings
- **Graph Builder**: NetworkX for relationships (planned)
- **Rule Engine**: Conditional logic extraction (planned)

### Reasoning Engine
- **Math Engine**: SymPy for symbolic computation
- **Unit Converter**: Pint for dimensional analysis
- **Calculator**: Step-by-step calculation tracking

### Query Processing
- **Intent Parser**: GPT-4 for understanding queries
- **Query Planner**: Multi-step execution planning
- **Response Generator**: Natural language answers with citations

## Example Queries

- "How much ventilation is required for a 6-car carpark?"
- "What is the minimum clearance for emergency exits?"
- "Calculate the total heat load for a 500 sqm office space"
- "Compare Class A and Class B fire resistance requirements"
- "Is 2.5m ceiling height compliant for a retail space?"

## Configuration

Edit `configs/config.yaml` to customize:
- OpenAI model settings
- Document processing parameters

## 🌟 Features

- **🕸️ Hybrid Search**: Vector + Graph database for comprehensive results
- **🧮 Mathematical Processing**: SymPy integration for calculations
- **📊 Table Extraction**: Camelot-based structured data extraction
- **🎨 Cyberpunk UI**: Futuristic interface with animations
- **🔗 Relationship Mapping**: NetworkX knowledge graphs
- **💬 Intelligent Chat**: GPT-4 powered conversations

## Configuration

Edit `configs/config.yaml` to customize:
- OpenAI model settings
- Document processing parameters
- Vector store configuration
- Calculation precision
- UI preferences

## Development

### Project Structure
```
technical-doc-ai/
├── src/
│   ├── document_processing/  # PDF parsing and extraction
│   ├── knowledge_base/       # Vector store and graph
│   ├── reasoning/            # Math and unit conversion
│   ├── query_processing/     # NLP and response generation
│   └── main.py              # Main application
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── data/                     # Sample documents
└── logs/                     # Application logs
```

### Adding New Features

1. **Custom Units**: Add to `src/reasoning/unit_converter.py`
2. **Document Types**: Extend parsers in `document_processing/`
3. **Query Types**: Add to `QueryType` enum in query processor

## Troubleshooting

### Common Issues

1. **OCR not working**: Install Tesseract
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu
   sudo apt-get install tesseract-ocr
   ```

2. **Camelot installation issues**: Install system dependencies
   ```bash
   # macOS
   brew install ghostscript tcl-tk
   
   # Ubuntu
   sudo apt-get install ghostscript python3-tk
   ```

3. **Memory issues with large PDFs**: Adjust chunk size in config.yaml

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Web Interface Features

The Streamlit web application provides:

- **📤 Document Upload**: Drag and drop PDF files for processing
- **💬 Interactive Q&A**: Natural language query interface
- **🧮 Calculation Display**: Step-by-step mathematical reasoning
- **📖 Source Citations**: Automatic section and page references
- **📊 Analytics**: Document statistics and query insights
- **📝 Query History**: Track previous questions and answers
- **🎯 Confidence Scoring**: AI confidence assessment for each answer

## Roadmap

- [x] Web interface with Streamlit
- [x] Knowledge graph with NetworkX
- [x] Rule engine for conditional logic
- [x] Comprehensive test suite
- [ ] Knowledge graph visualization in UI
- [ ] Multi-document cross-referencing
- [ ] Export functionality (PDF reports)
- [ ] API endpoints with FastAPI
- [ ] Docker containerization