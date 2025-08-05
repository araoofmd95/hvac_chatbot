"""
PDF Parser Module
Handles extraction of text, structure, and metadata from PDF documents
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import PyPDF2
from loguru import logger

# Optional pytesseract import for OCR
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Pytesseract not available. OCR functionality will be disabled.")


@dataclass
class DocumentSection:
    """Represents a section within a document"""
    level: int
    number: str
    title: str
    content: str
    page_number: int
    parent_section: Optional[str] = None
    subsections: List['DocumentSection'] = None
    
    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []


@dataclass
class ParsedDocument:
    """Container for parsed document data"""
    title: str
    sections: List[DocumentSection]
    metadata: Dict[str, any]
    raw_text: str
    total_pages: int


class PDFParser:
    """Robust PDF parser with multiple extraction strategies"""
    
    def __init__(self):
        self.section_patterns = [
            # Standard numbering patterns
            r'^(\d+(?:\.\d+)*)\s+(.+?)$',  # 1.2.3 Title
            r'^([A-Z](?:\.\d+)*)\s+(.+?)$',  # A.1.2 Title
            r'^(Chapter\s+\d+)\s*[:\-]?\s*(.+?)$',  # Chapter 1: Title
            r'^(Section\s+\d+(?:\.\d+)*)\s*[:\-]?\s*(.+?)$',  # Section 1.2: Title
        ]
        
    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        Parse a PDF document and extract structured content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument object containing parsed data
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Parsing PDF: {pdf_path}")
        
        # Try pdfplumber first (better for complex layouts)
        try:
            document = self._parse_with_pdfplumber(pdf_path)
            if document and document.raw_text.strip():
                return document
        except Exception as e:
            logger.warning(f"pdfplumber parsing failed: {e}")
            
        # Fallback to PyPDF2
        try:
            document = self._parse_with_pypdf2(pdf_path)
            if document and document.raw_text.strip():
                return document
        except Exception as e:
            logger.warning(f"PyPDF2 parsing failed: {e}")
            
        # Last resort: OCR
        logger.info("Attempting OCR extraction")
        return self._parse_with_ocr(pdf_path)
        
    def _parse_with_pdfplumber(self, pdf_path: Path) -> ParsedDocument:
        """Parse PDF using pdfplumber (layout-aware)"""
        sections = []
        raw_text = ""
        metadata = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            # Extract metadata
            metadata = pdf.metadata or {}
            total_pages = len(pdf.pages)
            
            # Process each page
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text while preserving layout
                page_text = page.extract_text(layout=True) or ""
                raw_text += page_text + "\n"
                
                # Extract sections from this page
                page_sections = self._extract_sections(page_text, page_num)
                sections.extend(page_sections)
                
        # Build hierarchy
        sections = self._build_section_hierarchy(sections)
        
        # Extract title (first non-empty line or from metadata)
        title = metadata.get('Title', '') or self._extract_title(raw_text)
        
        return ParsedDocument(
            title=title,
            sections=sections,
            metadata=metadata,
            raw_text=raw_text,
            total_pages=total_pages
        )
        
    def _parse_with_pypdf2(self, pdf_path: Path) -> ParsedDocument:
        """Parse PDF using PyPDF2 (simpler but less accurate)"""
        sections = []
        raw_text = ""
        metadata = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            if pdf_reader.metadata:
                metadata = {k: v for k, v in pdf_reader.metadata.items()}
                
            total_pages = len(pdf_reader.pages)
            
            # Process each page
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                raw_text += page_text + "\n"
                
                # Extract sections
                page_sections = self._extract_sections(page_text, page_num)
                sections.extend(page_sections)
                
        # Build hierarchy
        sections = self._build_section_hierarchy(sections)
        
        # Extract title
        title = metadata.get('/Title', '') or self._extract_title(raw_text)
        
        return ParsedDocument(
            title=title,
            sections=sections,
            metadata=metadata,
            raw_text=raw_text,
            total_pages=total_pages
        )
        
    def _parse_with_ocr(self, pdf_path: Path) -> ParsedDocument:
        """Parse PDF using OCR (for scanned documents)"""
        if not TESSERACT_AVAILABLE:
            raise ValueError("OCR not available. Install pytesseract and tesseract system dependency.")
            
        sections = []
        raw_text = ""
        
        # Convert PDF pages to images and OCR them
        try:
            import pdf2image
            pages = pdf2image.convert_from_path(pdf_path)
            total_pages = len(pages)
            
            for page_num, page_img in enumerate(pages, 1):
                # Perform OCR
                page_text = pytesseract.image_to_string(page_img)
                raw_text += page_text + "\n"
                
                # Extract sections
                page_sections = self._extract_sections(page_text, page_num)
                sections.extend(page_sections)
                
        except ImportError:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            raise
            
        # Build hierarchy
        sections = self._build_section_hierarchy(sections)
        
        # Extract title
        title = self._extract_title(raw_text)
        
        return ParsedDocument(
            title=title,
            sections=sections,
            metadata={},
            raw_text=raw_text,
            total_pages=total_pages
        )
        
    def _extract_sections(self, text: str, page_num: int) -> List[DocumentSection]:
        """Extract sections from text using regex patterns"""
        sections = []
        lines = text.split('\n')
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches any section pattern
            section_match = None
            for pattern in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_match = match
                    break
                    
            if section_match:
                # Save previous section content if any
                if current_content and sections:
                    sections[-1].content = '\n'.join(current_content).strip()
                    current_content = []
                    
                # Create new section
                section_number = section_match.group(1)
                section_title = section_match.group(2).strip()
                level = self._determine_section_level(section_number)
                
                section = DocumentSection(
                    level=level,
                    number=section_number,
                    title=section_title,
                    content="",
                    page_number=page_num
                )
                sections.append(section)
            else:
                # Add to current content
                current_content.append(line)
                
        # Save final section content
        if current_content and sections:
            sections[-1].content = '\n'.join(current_content).strip()
            
        return sections
        
    def _determine_section_level(self, section_number: str) -> int:
        """Determine the hierarchical level of a section"""
        if 'Chapter' in section_number:
            return 1
        elif 'Section' in section_number:
            # Extract actual number
            match = re.search(r'\d+(?:\.\d+)*', section_number)
            if match:
                return match.group(0).count('.') + 2
            return 2
        else:
            # Count dots for standard numbering
            return section_number.count('.') + 1
            
    def _build_section_hierarchy(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Build parent-child relationships between sections"""
        if not sections:
            return sections
            
        # Stack to track parent sections at each level
        parent_stack = []
        root_sections = []
        
        for section in sections:
            # Find appropriate parent
            while parent_stack and parent_stack[-1].level >= section.level:
                parent_stack.pop()
                
            if parent_stack:
                # Add as child of current parent
                parent = parent_stack[-1]
                parent.subsections.append(section)
                section.parent_section = parent.number
            else:
                # Top-level section
                root_sections.append(section)
                
            # Add to stack for potential children
            parent_stack.append(section)
            
        return root_sections
        
    def _extract_title(self, text: str) -> str:
        """Extract document title from text"""
        lines = text.strip().split('\n')
        
        # Look for title in first few non-empty lines
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            if line and len(line) > 10 and not re.match(r'^\d+', line):
                # Likely a title
                return line
                
        return "Untitled Document"