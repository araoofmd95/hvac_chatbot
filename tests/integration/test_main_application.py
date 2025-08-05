"""
Integration tests for the main application
"""
import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from main import TechnicalDocumentAI


class TestTechnicalDocumentAI:
    """Integration test cases for TechnicalDocumentAI"""
    
    @pytest.fixture
    def mock_openai_key(self):
        """Mock OpenAI API key for testing"""
        return "test-api-key-12345"
        
    @pytest.fixture
    def ai_system(self, mock_openai_key):
        """Create a TechnicalDocumentAI instance for testing"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': mock_openai_key}):
            return TechnicalDocumentAI(openai_api_key=mock_openai_key)
            
    def test_initialization(self, ai_system):
        """Test that the system initializes properly"""
        assert ai_system.pdf_parser is not None
        assert ai_system.table_extractor is not None
        assert ai_system.formula_extractor is not None
        assert ai_system.hierarchy_builder is not None
        assert ai_system.vector_store is not None
        assert ai_system.math_engine is not None
        assert ai_system.unit_converter is not None
        assert ai_system.query_processor is not None
        
    def test_no_api_key_raises_error(self):
        """Test that missing API key raises an error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                TechnicalDocumentAI()
                
    @patch('src.document_processing.pdf_parser.PDFParser.parse')
    @patch('src.document_processing.table_extractor.TableExtractor.extract_tables')
    @patch('src.knowledge_base.vector_store.VectorStore.add_documents')
    def test_document_ingestion_workflow(self, mock_add_docs, mock_extract_tables, mock_parse, ai_system):
        """Test the document ingestion workflow"""
        # Mock parsed document
        mock_parsed_doc = Mock()
        mock_parsed_doc.title = "Test Building Code"
        mock_parsed_doc.sections = [
            Mock(number="3.1", title="Ventilation", level=2, content="Test content", page_number=10)
        ]
        mock_parsed_doc.total_pages = 20
        mock_parse.return_value = mock_parsed_doc
        
        # Mock tables
        mock_extract_tables.return_value = []
        
        # Mock vector store
        mock_add_docs.return_value = 5
        
        # Test ingestion
        doc_id = ai_system.ingest_document("test.pdf", "Test Document")
        
        # Verify calls were made
        mock_parse.assert_called_once_with("test.pdf")
        mock_extract_tables.assert_called_once_with("test.pdf")
        mock_add_docs.assert_called_once()
        
        # Verify document was stored
        assert doc_id in ai_system.documents
        doc_info = ai_system.documents[doc_id]
        assert doc_info['name'] == "Test Document"
        assert doc_info['title'] == "Test Building Code"
        
    @patch('src.query_processing.query_processor.QueryProcessor.parse_query')
    @patch('src.knowledge_base.vector_store.VectorStore.search')
    @patch('src.query_processing.query_processor.QueryProcessor.generate_response')
    def test_question_answering_workflow(self, mock_generate, mock_search, mock_parse, ai_system):
        """Test the question answering workflow"""
        # Mock query parsing
        mock_intent = Mock()
        mock_intent.query_type.value = "calculation"
        mock_intent.parameters = {"num_cars": 6}
        mock_intent.required_calculations = ["total = num_cars * 300"]
        mock_intent.to_dict.return_value = {"query_type": "calculation"}
        mock_parse.return_value = mock_intent
        
        # Mock search results
        mock_search_result = Mock()
        mock_search_result.content = "Ventilation rate: 300 m³/hour per car"
        mock_search_result.metadata = {"section_number": "3.2.1"}
        mock_search.return_value = [mock_search_result]
        
        # Mock response generation
        mock_response = Mock()
        mock_response.answer = "For 6 cars, you need 1800 m³/hour ventilation"
        mock_response.calculations = [{"expression": "6 * 300", "result": 1800}]
        mock_response.confidence = 0.9
        mock_response.supporting_evidence = []
        mock_response.citations = []
        mock_response.reasoning_steps = []
        mock_generate.return_value = mock_response
        
        # Test question answering
        question = "How much ventilation is required for a 6-car carpark?"
        answer = ai_system.answer_question(question)
        
        # Verify workflow
        mock_parse.assert_called_once_with(question)
        mock_search.assert_called_once()
        mock_generate.assert_called_once()
        
        # Verify answer structure
        assert answer['question'] == question
        assert answer['answer'] == "For 6 cars, you need 1800 m³/hour ventilation"
        assert answer['confidence'] == 0.9
        
    def test_list_documents(self, ai_system):
        """Test listing documents"""
        # Initially empty
        docs = ai_system.list_documents()
        assert len(docs) == 0
        
        # Add a mock document
        ai_system.documents['test_id'] = {
            'id': 'test_id',
            'name': 'Test Doc',
            'title': 'Test Title'
        }
        
        docs = ai_system.list_documents()
        assert len(docs) == 1
        assert docs[0]['name'] == 'Test Doc'
        
    def test_get_document_info(self, ai_system):
        """Test getting document information"""
        # Non-existent document
        info = ai_system.get_document_info('nonexistent')
        assert info is None
        
        # Add and retrieve document
        ai_system.documents['test_id'] = {
            'id': 'test_id',
            'name': 'Test Doc'
        }
        
        info = ai_system.get_document_info('test_id')
        assert info is not None
        assert info['name'] == 'Test Doc'
        
    @patch('src.knowledge_base.vector_store.VectorStore.delete_collection')
    def test_clear_knowledge_base(self, mock_delete, ai_system):
        """Test clearing the knowledge base"""
        # Add some documents
        ai_system.documents['test1'] = {'id': 'test1'}
        ai_system.documents['test2'] = {'id': 'test2'}
        
        # Clear
        ai_system.clear_knowledge_base()
        
        # Verify
        mock_delete.assert_called_once()
        assert len(ai_system.documents) == 0
        
    def test_error_handling_in_question_answering(self, ai_system):
        """Test error handling in question answering"""
        with patch.object(ai_system.query_processor, 'parse_query', side_effect=Exception("Test error")):
            answer = ai_system.answer_question("Test question")
            
            assert 'error' in answer
            assert "Test error" in answer['answer']
            
    @patch('src.reasoning.math_engine.MathEngine.evaluate')
    def test_calculation_integration(self, mock_evaluate, ai_system):
        """Test integration with math engine for calculations"""
        # Mock calculation result
        mock_result = Mock()
        mock_result.numeric_result = 1800
        mock_result.error = None
        mock_result.steps = ["Step 1", "Step 2"]
        mock_evaluate.return_value = mock_result
        
        # Mock intent with calculation
        mock_intent = Mock()
        mock_intent.query_type.value = "calculation"
        mock_intent.required_calculations = ["num_cars * 300"]
        mock_intent.parameters = {"num_cars": 6}
        
        # Test calculation
        calculations = ai_system._perform_calculations(mock_intent, [])
        
        assert len(calculations) == 1
        assert calculations[0]['numeric_result'] == 1800
        mock_evaluate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])