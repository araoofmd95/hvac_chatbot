"""
Vector Store Module
Manages document embeddings and semantic search using ChromaDB and OpenAI
"""
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import uuid

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from loguru import logger
import openai
from openai import OpenAI


@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    content: str
    metadata: Dict[str, any]
    chunk_id: str
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Result from vector similarity search"""
    content: str
    metadata: Dict[str, any]
    score: float
    chunk_id: str


class VectorStore:
    """ChromaDB-based vector store with OpenAI embeddings"""
    
    def __init__(self, 
                 collection_name: str = "technical_docs",
                 persist_directory: str = "./chroma_db",
                 openai_api_key: Optional[str] = None):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            openai_api_key: OpenAI API key for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Set up OpenAI
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize ChromaDB
        self._init_chromadb()
        
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        # Create ChromaDB client with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection with OpenAI embeddings
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name="text-embedding-3-small"
        )
        
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        except Exception:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            
    def _sanitize_metadata(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """
        Sanitize metadata to ensure ChromaDB compatibility
        ChromaDB only accepts str, int, float, bool, or None as metadata values
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string
                sanitized[key] = ', '.join(str(v) for v in value)
            elif isinstance(value, dict):
                # Convert dict to JSON string
                sanitized[key] = json.dumps(value)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized
    
    def add_documents(self, chunks: List[DocumentChunk]) -> int:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of document chunks to add
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
            
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)
            # Sanitize metadata before adding
            sanitized_metadata = self._sanitize_metadata(chunk.metadata)
            metadatas.append(sanitized_metadata)
            
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
            
    def search(self, 
              query: str, 
              top_k: int = 10,
              filter_dict: Optional[Dict[str, any]] = None) -> List[SearchResult]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters to apply
            
        Returns:
            List of search results
        """
        try:
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_dict
            )
            
            # Parse results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = SearchResult(
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        score=1 - results['distances'][0][i],  # Convert distance to similarity
                        chunk_id=results['ids'][0][i]
                    )
                    search_results.append(result)
                    
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
            
    def get_by_ids(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """
        Retrieve specific chunks by their IDs
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
            
        Returns:
            List of document chunks
        """
        try:
            results = self.collection.get(ids=chunk_ids)
            
            chunks = []
            for i in range(len(results['ids'])):
                chunk = DocumentChunk(
                    content=results['documents'][i],
                    metadata=results['metadatas'][i],
                    chunk_id=results['ids'][i]
                )
                chunks.append(chunk)
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
            
    def delete_collection(self):
        """Delete the current collection"""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            
    def get_collection_stats(self) -> Dict[str, any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
            
    def create_chunks(self, 
                     text: str, 
                     metadata: Dict[str, any],
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200) -> List[DocumentChunk]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap
            
        Returns:
            List of document chunks
        """
        chunks = []
        
        # Simple sliding window chunking
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + chunk_size
            
            # Find a good breaking point (end of sentence/paragraph)
            if end < len(text):
                # Look for sentence end
                for sep in ['\n\n', '\n', '. ', '! ', '? ']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + len(sep)
                        break
                        
            # Create chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'chunk_start': start,
                    'chunk_end': end
                })
                
                # Generate unique chunk ID using UUID to avoid conflicts
                unique_suffix = str(uuid.uuid4())[:8]
                section_num = metadata.get('section_number', 'unknown')
                
                chunk_id_parts = [
                    metadata.get('doc_id', 'unknown'),
                    f"sec_{section_num}",
                    f"chunk_{chunk_index}",
                    unique_suffix
                ]
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id="_".join(chunk_id_parts)
                )
                chunks.append(chunk)
                chunk_index += 1
                
            # Move to next chunk
            start = end - chunk_overlap if end < len(text) else end
            
        return chunks
        
    def search_with_context(self, 
                           query: str,
                           top_k: int = 5,
                           context_chunks: int = 1) -> List[Tuple[SearchResult, List[DocumentChunk]]]:
        """
        Search and include surrounding context chunks
        
        Args:
            query: Search query
            top_k: Number of results
            context_chunks: Number of chunks before/after to include
            
        Returns:
            List of (result, context_chunks) tuples
        """
        # Get initial results
        results = self.search(query, top_k)
        
        results_with_context = []
        for result in results:
            # Extract chunk index from metadata
            chunk_index = result.metadata.get('chunk_index', 0)
            doc_id = result.metadata.get('doc_id', '')
            
            # Build list of context chunk IDs
            context_ids = []
            for i in range(-context_chunks, context_chunks + 1):
                if i == 0:
                    continue  # Skip the main chunk
                context_id = f"{doc_id}_{chunk_index + i}"
                context_ids.append(context_id)
                
            # Retrieve context chunks
            context = self.get_by_ids(context_ids)
            
            results_with_context.append((result, context))
            
        return results_with_context