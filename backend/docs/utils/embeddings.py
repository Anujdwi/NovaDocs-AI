import os
from typing import List

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '').strip()
AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY', '').strip()
AZURE_EMBEDDINGS_MODEL = os.getenv('AZURE_EMBEDDINGS_MODEL', 'text-embedding-3-small')

def _mock_embedding(text:str, dim=153) -> List[float]:
    h = abs(hash(text)) % (10**8)
    return [((h + i) % 100) / 100.0 for i in range(dim)]

def get_embedding_for_text(text:str) -> List[float]:
    """
    Return enbedding vector as list[float].
    If Azure OpenAI is not set this returns a deterministic mock vector (for local dev).
    """
    if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
        return _mock_embedding(text)
    
    try:
        from azure.ai.openai import AzureOpenAI  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("Install azure-ai-openai to call Azure OpenAI APIs: pip install azure-ai-openai") from e
    
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version="2024-02-15-preview",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )
    response = client.embeddings.create(model=AZURE_EMBEDDINGS_MODEL, input=text)
    return response.data[0].embedding