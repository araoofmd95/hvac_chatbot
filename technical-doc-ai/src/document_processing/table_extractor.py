"""
Table Extractor Module
Extracts and parses tables from PDF documents using Camelot
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from loguru import logger

# Optional camelot import
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("Camelot not available. Table extraction will be limited.")


@dataclass
class ExtractedTable:
    """Represents an extracted table from a document"""
    table_id: str
    page_number: int
    df: pd.DataFrame
    title: Optional[str] = None
    caption: Optional[str] = None
    headers: List[str] = None
    table_type: str = "data"  # data, lookup, reference
    metadata: Dict[str, any] = None
    
    def __post_init__(self):
        if self.headers is None and not self.df.empty:
            self.headers = list(self.df.columns)
        if self.metadata is None:
            self.metadata = {}
            
    def to_dict(self) -> Dict[str, any]:
        """Convert table to dictionary format"""
        return {
            'table_id': self.table_id,
            'page_number': self.page_number,
            'title': self.title,
            'caption': self.caption,
            'headers': self.headers,
            'table_type': self.table_type,
            'data': self.df.to_dict('records'),
            'metadata': self.metadata
        }
        
    def to_markdown(self) -> str:
        """Convert table to markdown format"""
        md = ""
        if self.title:
            md += f"### {self.title}\n\n"
        if self.caption:
            md += f"*{self.caption}*\n\n"
        md += self.df.to_markdown(index=False)
        return md


class TableExtractor:
    """Extracts tables from PDF documents with advanced parsing"""
    
    def __init__(self):
        self.extraction_methods = ['lattice', 'stream']
        
    def extract_tables(self, pdf_path: str, pages: str = 'all') -> List[ExtractedTable]:
        """
        Extract all tables from a PDF document
        
        Args:
            pdf_path: Path to PDF file
            pages: Page numbers to process ('all' or '1,2,3' or '1-3')
            
        Returns:
            List of extracted tables
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        logger.info(f"Extracting tables from: {pdf_path}")
        
        if not CAMELOT_AVAILABLE:
            logger.warning("Camelot not available. Skipping table extraction.")
            return []
        
        all_tables = []
        
        # Try both extraction methods
        for method in self.extraction_methods:
            try:
                tables = self._extract_with_method(pdf_path, method, pages)
                all_tables.extend(tables)
            except Exception as e:
                logger.warning(f"Table extraction failed with {method}: {e}")
                
        # Deduplicate tables
        unique_tables = self._deduplicate_tables(all_tables)
        
        # Post-process tables
        processed_tables = []
        for table in unique_tables:
            processed = self._process_table(table)
            if processed:
                processed_tables.append(processed)
                
        logger.info(f"Extracted {len(processed_tables)} tables")
        return processed_tables
        
    def _extract_with_method(self, pdf_path: Path, method: str, pages: str) -> List[ExtractedTable]:
        """Extract tables using specific method"""
        tables_list = []
        
        try:
            # Read tables with Camelot
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages,
                flavor=method,
                suppress_stdout=True
            )
            
            # Convert to ExtractedTable objects
            for idx, table in enumerate(tables):
                # Get page number
                page_num = table.page
                
                # Convert to dataframe
                df = table.df
                
                # Skip empty tables
                if df.empty or df.shape[0] < 2:
                    continue
                    
                # Create ExtractedTable
                extracted = ExtractedTable(
                    table_id=f"table_{page_num}_{idx}_{method}",
                    page_number=page_num,
                    df=df,
                    metadata={
                        'extraction_method': method,
                        'accuracy': table.accuracy,
                        'whitespace': table.whitespace
                    }
                )
                
                tables_list.append(extracted)
                
        except Exception as e:
            logger.error(f"Error with {method} extraction: {e}")
            
        return tables_list
        
    def _deduplicate_tables(self, tables: List[ExtractedTable]) -> List[ExtractedTable]:
        """Remove duplicate tables based on content similarity"""
        if len(tables) <= 1:
            return tables
            
        unique_tables = []
        seen_hashes = set()
        
        for table in tables:
            # Create content hash
            content_hash = self._get_table_hash(table.df)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_tables.append(table)
                
        return unique_tables
        
    def _get_table_hash(self, df: pd.DataFrame) -> str:
        """Create hash of table content"""
        # Convert dataframe to string representation
        content = df.to_csv(index=False, header=False)
        return str(hash(content))
        
    def _process_table(self, table: ExtractedTable) -> Optional[ExtractedTable]:
        """Process and clean extracted table"""
        df = table.df.copy()
        
        # Clean empty rows and columns
        df = self._clean_empty_cells(df)
        
        # Detect and set headers
        df, headers = self._detect_headers(df)
        table.df = df
        table.headers = headers
        
        # Detect table type
        table.table_type = self._detect_table_type(df)
        
        # Extract title and caption
        table.title = self._extract_table_title(df)
        
        # Handle merged cells
        df = self._handle_merged_cells(df)
        table.df = df
        
        # Validate table
        if not self._validate_table(df):
            return None
            
        return table
        
    def _clean_empty_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty rows and columns"""
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Replace NaN with empty string
        df = df.fillna('')
        
        return df
        
    def _detect_headers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Detect and extract table headers"""
        if df.empty:
            return df, []
            
        # Check if first row looks like headers
        first_row = df.iloc[0]
        
        # Heuristics for header detection
        is_header = True
        for val in first_row:
            val_str = str(val).strip()
            # Headers typically don't have numbers only
            if val_str.replace('.', '').replace(',', '').isdigit():
                is_header = False
                break
                
        if is_header:
            # Use first row as headers
            headers = [str(x).strip() for x in first_row]
            df.columns = headers
            df = df.iloc[1:].reset_index(drop=True)
            return df, headers
        else:
            # Generate generic headers
            headers = [f"Column_{i}" for i in range(len(df.columns))]
            df.columns = headers
            return df, headers
            
    def _detect_table_type(self, df: pd.DataFrame) -> str:
        """Detect the type of table"""
        if df.empty:
            return "unknown"
            
        # Check for lookup table characteristics
        if len(df.columns) == 2:
            # Likely a key-value lookup table
            return "lookup"
            
        # Check for reference table (lots of text)
        if hasattr(df, 'map'):
            text_ratio = sum(df.map(lambda x: isinstance(x, str) and len(str(x)) > 20).sum()) / df.size
        else:
            text_ratio = sum(df.applymap(lambda x: isinstance(x, str) and len(str(x)) > 20).sum()) / df.size
        if text_ratio > 0.5:
            return "reference"
            
        # Default to data table
        return "data"
        
    def _extract_table_title(self, df: pd.DataFrame) -> Optional[str]:
        """Try to extract table title from surrounding context"""
        # This would require access to the text around the table
        # For now, return None
        return None
        
    def _handle_merged_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle merged cells in tables"""
        # Forward fill empty cells that might be merged
        for col in df.columns:
            # Only forward fill if pattern suggests merged cells
            col_data = df[col].astype(str)
            empty_count = (col_data == '').sum()
            
            if empty_count > 0 and empty_count < len(df) * 0.5:
                # Forward fill empty cells
                df[col] = df[col].replace('', pd.NA).fillna(method='ffill').fillna('')
                
        return df
        
    def _validate_table(self, df: pd.DataFrame) -> bool:
        """Validate if extracted table is valid"""
        # Must have at least 1 row
        if len(df) < 1:
            return False
            
        # Must have at least 2 columns
        if len(df.columns) < 2:
            return False
            
        # Check if table has any content
        try:
            # Use map instead of deprecated applymap
            if hasattr(df, 'map'):
                non_empty = df.map(lambda x: str(x).strip() != '').sum().sum()
            else:
                non_empty = df.applymap(lambda x: str(x).strip() != '').sum().sum()
            
            if non_empty < df.size * 0.2:  # Less than 20% non-empty
                return False
        except Exception:
            # If we can't validate content, assume it's valid
            return True
            
        return True
        
    def extract_lookup_values(self, table: ExtractedTable, key: str) -> Optional[str]:
        """Extract value from a lookup table"""
        if table.table_type != "lookup" or len(table.df.columns) < 2:
            return None
            
        df = table.df
        key_col = df.columns[0]
        value_col = df.columns[1]
        
        # Search for key
        mask = df[key_col].astype(str).str.contains(key, case=False, na=False)
        matches = df[mask]
        
        if not matches.empty:
            return str(matches.iloc[0][value_col])
            
        return None
        
    def search_tables(self, tables: List[ExtractedTable], query: str) -> List[ExtractedTable]:
        """Search for tables containing specific content"""
        matching_tables = []
        
        for table in tables:
            try:
                # Search in dataframe - use map instead of deprecated applymap
                if hasattr(table.df, 'map'):
                    mask = table.df.map(
                        lambda x: query.lower() in str(x).lower()
                    ).any(axis=1)
                else:
                    mask = table.df.applymap(
                        lambda x: query.lower() in str(x).lower()
                    ).any(axis=1)
                
                # Check if any row matches
                if mask.any():
                    matching_tables.append(table)
            except Exception:
                # If search fails, skip this table
                continue
                
        return matching_tables