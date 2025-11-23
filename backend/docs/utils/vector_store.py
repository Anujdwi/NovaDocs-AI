"""
ChromaDB Vector Store Integration
Handles storing and retrieving document chunks from ChromaDB for semantic search.
"""
import os
from typing import List, Dict, Optional
from django.conf import settings

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


# ChromaDB connection settings
# Default to 'chroma' for Docker, but allow override for local development
CHROMA_HOST = os.getenv('CHROMA_HOST', 'chroma')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION', 'document_chunks')


class ChromaVectorStore:
    """Wrapper for ChromaDB operations."""
    
    def __init__(self):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        
        try:
            self.client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            # Test connection
            self.client.heartbeat()
            self.collection = self._get_or_create_collection()
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}. "
                f"Make sure ChromaDB is running. Error: {str(e)}"
            ) from e
    
    def _get_or_create_collection(self):
        """Get or create the document chunks collection."""
        try:
            collection = self.client.get_collection(name=COLLECTION_NAME)
        except Exception:
            # Collection doesn't exist, create it
            collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Document chunks for RAG system"}
            )
        return collection
    
    def add_chunks(self, chunks: List[Dict]):
        """
        Add chunks to ChromaDB.
        
        Args:
            chunks: List of dicts with keys:
                - chunk_id: str
                - chunk_text: str
                - embedding: List[float]
                - metadata: dict (optional)
        """
        if not chunks:
            return
        
        ids = [c['chunk_id'] for c in chunks]
        texts = [c['chunk_text'] for c in chunks]
        embeddings = [c['embedding'] for c in chunks]
        metadatas = []
        
        for c in chunks:
            meta = c.get('metadata', {})
            # Add document reference if available
            if 'doc_id' in c:
                meta['doc_id'] = str(c['doc_id'])
            if 'doc_title' in c:
                meta['doc_title'] = c['doc_title']
            metadatas.append(meta)
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar chunks using query embedding.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"doc_id": "123"})
        
        Returns:
            List of dicts with keys: chunk_id, chunk_text, metadata, distance
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict if filter_dict else None
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'chunk_id': results['ids'][0][i],
                    'chunk_text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        return formatted_results
    
    def delete_chunks_by_doc_id(self, doc_id: str):
        """Delete all chunks for a specific document."""
        # ChromaDB doesn't support direct deletion by metadata filter in all versions
        # This is a simplified version - in production, you might want to track IDs
        try:
            # Get all chunks for this doc
            results = self.collection.get(
                where={"doc_id": str(doc_id)}
            )
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            print(f"Error deleting chunks for doc {doc_id}: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            'collection_name': COLLECTION_NAME,
            'total_chunks': count
        }


def get_vector_store() -> ChromaVectorStore:
    """Factory function to get a ChromaVectorStore instance."""
    return ChromaVectorStore()

