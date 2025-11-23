"""
RAG (Retrieval-Augmented Generation) Engine
Handles query processing: embedding query, retrieving context, and generating answers.
"""
import os
from typing import List, Dict, Optional
from django.conf import settings
from docs.utils.embeddings import get_embedding_for_text
from docs.utils.vector_store import get_vector_store

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '').strip()
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY', '').strip()
AZURE_LLM_MODEL = os.getenv('AZURE_LLM_MODEL', 'gpt-4o')
AZURE_LLM_DEPLOYMENT = os.getenv('AZURE_LLM_DEPLOYMENT', 'gpt-4o')


class RAGEngine:
    """RAG engine for processing queries and generating answers."""
    
    def __init__(self):
        try:
            self.vector_store = get_vector_store()
        except Exception as e:
            # Store error for later - don't fail on init, fail on query
            self.vector_store = None
            self._init_error = str(e)
        self.top_k = int(os.getenv('RAG_TOP_K', '5'))
        self.max_context_length = int(os.getenv('RAG_MAX_CONTEXT', '3000'))
    
    def _build_rag_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Build the RAG prompt with context and question."""
        context_text = "\n\n---\n\n".join([
            f"[Document: {chunk.get('metadata', {}).get('doc_title', 'Unknown')}]\n{chunk['chunk_text']}"
            for chunk in context_chunks
        ])
        
        # Truncate if too long
        if len(context_text) > self.max_context_length:
            context_text = context_text[:self.max_context_length] + "..."
        
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from documents.

[CONTEXT]
{context_text}

[QUESTION]
{query}

[INSTRUCTIONS]
- Answer the question based ONLY on the provided context.
- If the context doesn't contain enough information, say so.
- Cite which document(s) you used in your answer.
- Be concise and accurate.

[ANSWER]
"""
        return prompt
    
    def _call_azure_openai(self, prompt: str) -> str:
        """Call Azure OpenAI to generate answer."""
        if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
            # Mock response for development
            return "This is a placeholder answer. Please configure AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT to get real answers."
        
        try:
            from azure.ai.openai import AzureOpenAI  # type: ignore[import-untyped]
        except ImportError as e:
            return f"Azure OpenAI SDK not installed: {str(e)}. Install with: pip install azure-ai-openai"
        
        try:
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                api_version="2024-02-15-preview",
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            
            response = client.chat.completions.create(
                model=AZURE_LLM_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error calling Azure OpenAI: {str(e)}"
    
    def handle_query(self, query: str, filter_by_doc_id: Optional[str] = None) -> Dict:
        """
        Process a query through the RAG pipeline.
        
        Args:
            query: User's question
            filter_by_doc_id: Optional document ID to filter search
        
        Returns:
            Dict with keys: answer, sources, chunks_used
        """
        if self.vector_store is None:
            return {
                'answer': f"Vector store not available: {getattr(self, '_init_error', 'Unknown error')}. "
                         f"Please ensure ChromaDB is running and accessible.",
                'sources': [],
                'chunks_used': 0
            }
        
        # Step 1: Convert query to embedding
        query_embedding = get_embedding_for_text(query)
        
        # Step 2: Retrieve relevant chunks from ChromaDB
        try:
            filter_dict = {"doc_id": filter_by_doc_id} if filter_by_doc_id else None
            retrieved_chunks = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=self.top_k,
                filter_dict=filter_dict
            )
        except Exception as e:
            return {
                'answer': f"Error retrieving chunks from vector store: {str(e)}",
                'sources': [],
                'chunks_used': 0
            }
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                'sources': [],
                'chunks_used': 0
            }
        
        # Step 3: Build RAG prompt
        prompt = self._build_rag_prompt(query, retrieved_chunks)
        
        # Step 4: Generate answer using Azure OpenAI
        answer = self._call_azure_openai(prompt)
        
        # Step 5: Extract sources
        sources = []
        seen_docs = set()
        for chunk in retrieved_chunks:
            doc_id = chunk.get('metadata', {}).get('doc_id')
            doc_title = chunk.get('metadata', {}).get('doc_title', 'Unknown Document')
            if doc_id and doc_id not in seen_docs:
                sources.append({
                    'doc_id': doc_id,
                    'doc_title': doc_title,
                    'chunk_id': chunk['chunk_id']
                })
                seen_docs.add(doc_id)
        
        return {
            'answer': answer,
            'sources': sources,
            'chunks_used': len(retrieved_chunks)
        }


def get_rag_engine() -> RAGEngine:
    """Factory function to get a RAGEngine instance."""
    return RAGEngine()

