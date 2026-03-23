import asyncio
import logging
from pinecone import Pinecone
from core.config import settings

logger = logging.getLogger(__name__)

try:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(name=settings.PINECONE_INDEX_NAME)
except Exception as e:
    logger.error("Error when connecting to Pinecone!")

async def upsert_to_pinecone_batch(vectors_batch: list[tuple], namespace: str = "career-portfolio") -> bool:
    """
    Upsert the data into Pinecone Database
    vectors_batch: tuple (id, vector, metadata)
    """

    try:
        await asyncio.to_thread(
            index.upsert,
            vectors=vectors_batch,
            namespace=namespace
        )
        return True
    except Exception as e:
        logger.error("Error when upserting to Pinecone!")
        return False