from openai import AsyncOpenAI
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Generate the open AI client
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def get_embedding(text: str):
    """
    Embed the text into embeddings
    """
    # Clean the text
    text = text.replace("\n", " ")
    
    try:
        response = await client.embeddings.create(
            input=[text],
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS
        )
        return response.data[0].embedding
        
    except Exception as e:
        logger.error(f"Error when creating embeddings: {str(e)}")
        return None

async def get_embeddings_batch(texts: list[str]):
    """
    Creating embeddings for batch of data. Optimizing performance 
    and optimizing calling API.
    """
    try:
        response = await client.embeddings.create(
            input=texts,
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Error when creating embeddings: {str(e)}")
        return []