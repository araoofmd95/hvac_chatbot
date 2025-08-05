"""
Technical Document AI - Main Application
Integrates all components for intelligent document processing and Q&A
"""
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import uuid

from dotenv import load_dotenv
from loguru import logger

# Document processing
from document_processing.pdf_parser import PDFParser
from document_processing.table_extractor import TableExtractor
from document_processing.formula_extractor import FormulaExtractor
from document_processing.hierarchy_builder import HierarchyBuilder

# Knowledge base
from knowledge_base.vector_store import VectorStore, DocumentChunk
from knowledge_base.graph_builder import GraphBuilder

# Reasoning
from reasoning.math_engine import MathEngine
from reasoning.unit_converter import UnitConverter

# Query processing
from query_processing.query_processor import QueryProcessor, QueryType


class TechnicalDocumentAI:
    """Main application class integrating all components"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the Technical Document AI system
        
        Args:
            openai_api_key: OpenAI API key (will use env var if not provided)
        """
        # Load environment variables
        load_dotenv()
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        # Initialize components
        logger.info("Initializing Technical Document AI components...")
        
        # Document processors
        self.pdf_parser = PDFParser()
        self.table_extractor = TableExtractor()
        self.formula_extractor = FormulaExtractor()
        self.hierarchy_builder = HierarchyBuilder()
        
        # Knowledge base
        self.vector_store = VectorStore(
            collection_name="technical_docs",
            openai_api_key=self.openai_api_key
        )
        self.graph_builder = GraphBuilder()
        
        # Reasoning engines
        self.math_engine = MathEngine()
        self.unit_converter = UnitConverter()
        
        # Query processor
        self.query_processor = QueryProcessor(
            openai_api_key=self.openai_api_key
        )
        
        # Document metadata storage
        self.documents = {}
        
        logger.info("Technical Document AI initialized successfully")
        
    def ingest_document(self, pdf_path: str, doc_name: Optional[str] = None) -> str:
        """
        Ingest a PDF document into the system
        
        Args:
            pdf_path: Path to the PDF file
            doc_name: Optional name for the document
            
        Returns:
            Document ID
        """
        logger.info(f"Ingesting document: {pdf_path}")
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        if not doc_name:
            doc_name = Path(pdf_path).stem
            
        try:
            # Validate file
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            file_size_mb = pdf_file.stat().st_size / 1024 / 1024
            if file_size_mb > 100:  # 100MB limit
                raise ValueError(f"File too large: {file_size_mb:.1f}MB (max 100MB)")
            
            logger.info(f"Processing PDF: {pdf_file.name} ({file_size_mb:.1f}MB)")
            
            # Parse PDF
            logger.info("Parsing PDF structure...")
            try:
                parsed_doc = self.pdf_parser.parse(pdf_path)
                logger.info(f"PDF parsed: {len(parsed_doc.sections)} sections, {parsed_doc.total_pages} pages")
            except Exception as e:
                logger.error(f"PDF parsing failed: {e}")
                raise ValueError(f"Failed to parse PDF. The file might be corrupted, password-protected, or in an unsupported format: {str(e)}")
            
            # Extract tables (with error handling)
            logger.info("Extracting tables...")
            tables = []
            try:
                tables = self.table_extractor.extract_tables(pdf_path)
                logger.info(f"Extracted {len(tables)} tables")
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
                logger.info("Continuing without table extraction...")
            
            # Extract formulas
            logger.info("Extracting formulas...")
            formulas = []
            try:
                for section in parsed_doc.sections:
                    section_formulas = self.formula_extractor.extract_formulas(
                        section.content,
                        {
                            'section_number': section.number,
                            'section_title': section.title,
                            'page_number': section.page_number
                        }
                    )
                    formulas.extend(section_formulas)
                logger.info(f"Extracted {len(formulas)} formulas")
            except Exception as e:
                logger.warning(f"Formula extraction failed: {e}")
                logger.info("Continuing without formula extraction...")
                
            # Build hierarchy
            logger.info("Building document hierarchy...")
            section_data = [
                {
                    'number': s.number,
                    'title': s.title,
                    'level': s.level,
                    'content': s.content,
                    'page_number': s.page_number
                }
                for s in parsed_doc.sections
            ]
            hierarchy_nodes, section_lookup = self.hierarchy_builder.build_hierarchy(section_data)
            
            # Build knowledge graph
            logger.info("Building knowledge graph...")
            try:
                self._build_knowledge_graph(
                    doc_id, parsed_doc, tables, formulas, hierarchy_nodes
                )
                logger.info("Knowledge graph built successfully")
            except Exception as e:
                logger.warning(f"Knowledge graph building failed: {e}")
                logger.info("Continuing without knowledge graph...")
            
            # Create chunks for vector store
            logger.info("Creating document chunks...")
            chunks = self._create_document_chunks(
                parsed_doc, tables, formulas, doc_id, doc_name
            )
            
            # Add to vector store
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            try:
                added_count = self.vector_store.add_documents(chunks)
                logger.info(f"Successfully added {added_count} chunks to vector store")
            except Exception as e:
                logger.error(f"Vector store operation failed: {e}")
                raise ValueError(f"Failed to add document to knowledge base: {str(e)}")
            
            # Store document metadata
            self.documents[doc_id] = {
                'id': doc_id,
                'name': doc_name,
                'path': pdf_path,
                'title': parsed_doc.title,
                'sections': len(parsed_doc.sections),
                'tables': len(tables),
                'formulas': len(formulas),
                'total_pages': parsed_doc.total_pages,
                'hierarchy': hierarchy_nodes,
                'section_lookup': section_lookup
            }
            
            logger.info(f"Document ingested successfully. ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
            
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question about the ingested documents
        
        Args:
            question: Natural language question
            
        Returns:
            Dictionary containing answer and supporting information
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Parse query intent
            intent = self.query_processor.parse_query(question)
            logger.info(f"Query type: {intent.query_type.value}")
            
            # Retrieve relevant context
            context = self._retrieve_context(question, intent)
            
            # Perform calculations if needed
            calculations = None
            if intent.query_type == QueryType.CALCULATION:
                calculations = self._perform_calculations(intent, context)
                
            # Generate response
            response = self.query_processor.generate_response(
                question, intent, context, calculations
            )
            
            # Format final answer
            return {
                'question': question,
                'answer': response.answer,
                'intent': intent.to_dict(),
                'supporting_evidence': response.supporting_evidence,
                'calculations': response.calculations,
                'citations': response.citations,
                'confidence': response.confidence,
                'reasoning_steps': response.reasoning_steps
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                'question': question,
                'answer': f"I encountered an error processing your question: {str(e)}",
                'error': str(e)
            }
            
    def _create_document_chunks(self, 
                              parsed_doc,
                              tables,
                              formulas,
                              doc_id: str,
                              doc_name: str) -> List[DocumentChunk]:
        """Create chunks from parsed document components"""
        chunks = []
        
        # Chunk sections
        for section in parsed_doc.sections:
            section_chunks = self.vector_store.create_chunks(
                section.content,
                {
                    'doc_id': doc_id,
                    'doc_name': doc_name,
                    'type': 'section',
                    'section_number': section.number,
                    'section_title': section.title,
                    'page_number': section.page_number,
                    'level': section.level
                }
            )
            chunks.extend(section_chunks)
            
        # Add tables as chunks
        for table in tables:
            table_text = f"Table on page {table.page_number}:\n{table.to_markdown()}"
            table_chunk = DocumentChunk(
                content=table_text,
                metadata={
                    'doc_id': doc_id,
                    'doc_name': doc_name,
                    'type': 'table',
                    'table_id': table.table_id,
                    'page_number': table.page_number,
                    'table_type': table.table_type
                },
                chunk_id=f"{doc_id}_table_{table.table_id}"
            )
            chunks.append(table_chunk)
            
        # Add formulas as chunks
        for formula in formulas:
            formula_text = f"Formula: {formula.raw_text}\nVariables: {', '.join(formula.variables)}"
            if formula.context:
                formula_text += f"\nContext: {formula.context}"
                
            formula_chunk = DocumentChunk(
                content=formula_text,
                metadata={
                    'doc_id': doc_id,
                    'doc_name': doc_name,
                    'type': 'formula',
                    'formula_id': formula.formula_id,
                    'formula_type': formula.formula_type.value,
                    'variables': formula.variables
                },
                chunk_id=f"{doc_id}_{formula.formula_id}"
            )
            chunks.append(formula_chunk)
            
        return chunks
        
    def _retrieve_context(self, question: str, intent) -> List[Dict[str, Any]]:
        """Retrieve relevant context using both vector search and graph relationships"""
        
        # Step 1: Get initial results from vector store
        search_results = self.vector_store.search(
            question,
            top_k=8  # Reduced to leave room for graph-enhanced context
        )
        
        # Convert to context format
        context = []
        relevant_node_ids = set()
        
        for result in search_results:
            context_item = {
                'content': result.content,
                'score': result.score,
                'source': 'vector_search',
                **result.metadata
            }
            context.append(context_item)
            
            # Track node IDs for graph expansion
            if 'doc_id' in result.metadata:
                doc_id = result.metadata['doc_id']
                if result.metadata.get('type') == 'section':
                    section_num = result.metadata.get('section_number')
                    if section_num:
                        relevant_node_ids.add(f"{doc_id}_section_{section_num}")
                elif result.metadata.get('type') == 'formula':
                    formula_id = result.metadata.get('formula_id')
                    if formula_id:
                        relevant_node_ids.add(f"{doc_id}_{formula_id}")
                elif result.metadata.get('type') == 'table':
                    table_id = result.metadata.get('table_id')
                    if table_id:
                        relevant_node_ids.add(f"{doc_id}_table_{table_id}")
        
        # Step 2: Expand context using graph relationships
        enhanced_context = self._expand_context_with_graph(context, relevant_node_ids)
        
        # Step 3: Filter by intent requirements
        if intent.required_sections:
            # Prioritize specific sections
            filtered = []
            for ctx in enhanced_context:
                section_title = ctx.get('section_title', '').lower()
                if any(req.lower() in section_title for req in intent.required_sections):
                    filtered.append(ctx)
            if filtered:
                enhanced_context = filtered + [c for c in enhanced_context if c not in filtered]
        
        # Step 4: Limit total context to prevent overwhelming the AI
        return enhanced_context[:15]  # Increased limit to accommodate graph expansion
        
    def _expand_context_with_graph(self, base_context: List[Dict[str, Any]], 
                                  relevant_node_ids: Set[str]) -> List[Dict[str, Any]]:
        """Expand context using graph relationships"""
        
        enhanced_context = base_context.copy()
        added_node_ids = set()
        
        # For each relevant node, find related nodes
        for node_id in relevant_node_ids:
            try:
                # Get related nodes (1-hop neighbors)
                related_nodes = self.graph_builder.get_related_nodes(
                    node_id, 
                    relationship_types=['contains', 'shares_variable', 'references'],
                    max_relationships=3
                )
                
                for related_node in related_nodes:
                    related_id = related_node.node_id
                    
                    # Skip if already added
                    if related_id in added_node_ids:
                        continue
                        
                    # Create context item from graph node
                    graph_context = self._create_context_from_graph_node(
                        related_node, related_id
                    )
                    
                    if graph_context:
                        enhanced_context.append(graph_context)
                        added_node_ids.add(related_id)
                        
            except Exception as e:
                logger.warning(f"Failed to expand context for node {node_id}: {e}")
                continue
        
        return enhanced_context
        
    def _create_context_from_graph_node(self, node, node_id: str) -> Optional[Dict[str, Any]]:
        """Create context item from graph node"""
        
        try:
            if node.node_type == 'section':
                return {
                    'content': node.properties.get('content_preview', ''),
                    'score': 0.7,  # Lower than vector search but still relevant
                    'source': 'graph_expansion',
                    'type': 'section',
                    'section_number': node.properties.get('section_number'),
                    'section_title': node.label,
                    'page_number': node.properties.get('page_number'),
                    'expansion_reason': 'related_section'
                }
            elif node.node_type == 'formula':
                return {
                    'content': f"Formula: {node.label}\nVariables: {', '.join(node.properties.get('variables', []))}",
                    'score': 0.8,  # Higher relevance for formulas
                    'source': 'graph_expansion',
                    'type': 'formula',
                    'formula_id': node_id.split('_')[-1] if '_' in node_id else node_id,
                    'formula_type': node.properties.get('formula_type'),
                    'variables': node.properties.get('variables', []),
                    'expansion_reason': 'related_formula'
                }
            elif node.node_type == 'table':
                return {
                    'content': f"Table: {node.label} on page {node.properties.get('page_number', 'Unknown')}",
                    'score': 0.6,
                    'source': 'graph_expansion',
                    'type': 'table',
                    'table_id': node_id.split('_')[-1] if '_' in node_id else node_id,
                    'page_number': node.properties.get('page_number'),
                    'expansion_reason': 'related_table'
                }
        except Exception as e:
            logger.warning(f"Failed to create context from node {node_id}: {e}")
            
        return None
        
    def _perform_calculations(self, intent, context) -> List[Dict[str, Any]]:
        """Perform required calculations"""
        calculations = []
        
        for calc_expr in intent.required_calculations:
            try:
                # Extract variables from parameters
                variables = intent.parameters
                
                # Evaluate calculation
                result = self.math_engine.evaluate(calc_expr, variables)
                
                if not result.error:
                    calculations.append({
                        'expression': calc_expr,
                        'variables': variables,
                        'numeric_result': result.numeric_result,
                        'symbolic_result': str(result.symbolic_result),
                        'steps': result.steps
                    })
                else:
                    logger.error(f"Calculation error: {result.error}")
                    
            except Exception as e:
                logger.error(f"Error performing calculation: {e}")
                
        return calculations
        
    def _build_knowledge_graph(self, doc_id: str, parsed_doc, tables, formulas, hierarchy_nodes):
        """Build knowledge graph for the document"""
        
        # Add document root node
        self.graph_builder.add_document_node(doc_id, parsed_doc.title or "Unknown Document")
        
        # Add section nodes and relationships
        for section in parsed_doc.sections:
            section_id = f"{doc_id}_section_{section.number}"
            
            # Add section node
            self.graph_builder.add_section_node(
                section_id,
                section.title or f"Section {section.number}",
                {
                    'section_number': section.number,
                    'level': section.level,
                    'page_number': section.page_number,
                    'content_preview': section.content[:200] if section.content else ""
                }
            )
            
            # Link section to document
            self.graph_builder.add_relationship(
                doc_id, section_id, "contains", 
                {'relationship_type': 'hierarchical'}
            )
        
        # Add table nodes and relationships
        for table in tables:
            table_id = f"{doc_id}_table_{table.table_id}"
            
            # Add table node
            self.graph_builder.add_table_node(
                table_id,
                f"Table {table.table_id}",
                {
                    'table_type': table.table_type,
                    'page_number': table.page_number,
                    'column_count': len(table.columns) if hasattr(table, 'columns') else 0
                }
            )
            
            # Link table to document
            self.graph_builder.add_relationship(
                doc_id, table_id, "contains",
                {'relationship_type': 'structural'}
            )
            
            # Find which section contains this table (by page number)
            for section in parsed_doc.sections:
                if section.page_number == table.page_number:
                    section_id = f"{doc_id}_section_{section.number}"
                    self.graph_builder.add_relationship(
                        section_id, table_id, "contains",
                        {'relationship_type': 'spatial'}
                    )
                    break
        
        # Add formula nodes and relationships
        for formula in formulas:
            formula_id = f"{doc_id}_{formula.formula_id}"
            
            # Add formula node
            self.graph_builder.add_formula_node(
                formula_id,
                formula.raw_text,
                {
                    'formula_type': formula.formula_type.value,
                    'variables': formula.variables,
                    'operators': formula.operators,
                    'normalized_text': formula.normalized_text
                }
            )
            
            # Link formula to document
            self.graph_builder.add_relationship(
                doc_id, formula_id, "contains",
                {'relationship_type': 'mathematical'}
            )
            
            # Link formula to section based on source location
            if 'section_number' in formula.source_location:
                section_number = formula.source_location['section_number']
                section_id = f"{doc_id}_section_{section_number}"
                self.graph_builder.add_relationship(
                    section_id, formula_id, "contains",
                    {'relationship_type': 'contextual'}
                )
        
        # Build formula dependencies
        self._build_formula_dependencies(doc_id, formulas)
        
        # Build cross-references between sections
        self._build_section_references(doc_id, parsed_doc.sections)
        
    def _build_formula_dependencies(self, doc_id: str, formulas):
        """Build relationships between formulas based on shared variables"""
        
        # Group formulas by variables
        variable_to_formulas = {}
        for formula in formulas:
            for variable in formula.variables:
                if variable not in variable_to_formulas:
                    variable_to_formulas[variable] = []
                variable_to_formulas[variable].append(formula)
        
        # Create dependencies between formulas that share variables
        for variable, formula_list in variable_to_formulas.items():
            if len(formula_list) > 1:
                for i, formula1 in enumerate(formula_list):
                    for j, formula2 in enumerate(formula_list[i+1:], i+1):
                        formula1_id = f"{doc_id}_{formula1.formula_id}"
                        formula2_id = f"{doc_id}_{formula2.formula_id}"
                        
                        # Add bidirectional relationship
                        self.graph_builder.add_relationship(
                            formula1_id, formula2_id, "shares_variable",
                            {'shared_variable': variable}
                        )
                        self.graph_builder.add_relationship(
                            formula2_id, formula1_id, "shares_variable",
                            {'shared_variable': variable}
                        )
        
    def _build_section_references(self, doc_id: str, sections):
        """Build references between sections based on content analysis"""
        
        # Simple approach: look for section number references in content
        section_numbers = {section.number for section in sections}
        
        for section in sections:
            section_id = f"{doc_id}_section_{section.number}"
            content = section.content.lower() if section.content else ""
            
            # Look for references to other sections
            for other_section in sections:
                if other_section.number != section.number:
                    # Look for patterns like "Section 5.1", "see 5.1", "clause 5.1"
                    ref_patterns = [
                        f"section {other_section.number}",
                        f"clause {other_section.number}",
                        f"see {other_section.number}",
                        f"refer {other_section.number}"
                    ]
                    
                    for pattern in ref_patterns:
                        if pattern in content:
                            other_section_id = f"{doc_id}_section_{other_section.number}"
                            self.graph_builder.add_relationship(
                                section_id, other_section_id, "references",
                                {'reference_type': 'textual'}
                            )
                            break
        
    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an ingested document"""
        return self.documents.get(doc_id)
        
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents"""
        return list(self.documents.values())
        
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base"""
        logger.warning("Clearing knowledge base...")
        self.vector_store.delete_collection()
        self.graph_builder.clear_graph()
        self.documents = {}
        
        # Reinitialize vector store
        self.vector_store = VectorStore(
            collection_name="technical_docs",
            openai_api_key=self.openai_api_key
        )
        logger.info("Knowledge base cleared")


# Example usage
if __name__ == "__main__":
    # Initialize system
    ai_system = TechnicalDocumentAI()
    
    # Example: Ingest a document
    # doc_id = ai_system.ingest_document("path/to/building_code.pdf")
    
    # Example: Ask a question
    # answer = ai_system.answer_question("How much ventilation is required for a 6-car carpark?")
    # print(answer['answer'])