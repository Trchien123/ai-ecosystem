import os
import glob
import asyncio
import hashlib
import logging
import re
from pathlib import Path

from core.config import settings
from packages.shared_utils.shared_utils.embeddings import get_embeddings_batch
from packages.shared_utils.shared_utils.pinecone_client import upsert_to_pinecone_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - 🤖 %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

MAX_RETRIES = 3
BATCH_SIZE = 100


# CHUNKING: Split by markdown heading, fallback
# to word-overlap if the section is too long

def chunk_by_headings(text: str, max_words: int = 150, overlap_words: int = 30) -> list[str]:
    """
    Split on markdown headings (##, ###, …) to keep semantic units intact.
    If a section exceeds max_words, fall back to sliding-window word chunks
       so we never lose context across a heading boundary.
    """
    # Split on lines that start with one or more '#'
    sections = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[str] = []
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            chunks.append(section)
        else:
            # Sliding window fallback
            i = 0
            while i < len(words):
                chunk_words = words[i: i + max_words]
                chunks.append(" ".join(chunk_words))
                i += max_words - overlap_words

    return chunks


def stable_vector_id(filename: str, chunk_index: int, chunk_text: str) -> str:
    """
    Deterministic ID based on filename + local chunk index + a short content hash.
    Re-running with the same data produces the same IDs → safe to upsert repeatedly.
    """
    content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:8]
    safe_name = Path(filename).stem.replace(" ", "_")
    return f"{safe_name}-chunk-{chunk_index}-{content_hash}"


# Upsert with retry
async def upsert_with_retry(batch: list, namespace: str, retries: int = MAX_RETRIES) -> bool:
    for attempt in range(1, retries + 1):
        try:
            result = await upsert_to_pinecone_batch(batch, namespace=namespace)
            if result:
                return True
            logger.warning(f"Upsert returned falsy on attempt {attempt}.")
        except Exception as e:
            logger.warning(f"Upsert attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)  # exponential back-off
    logger.error(f"Batch permanently failed after {retries} attempts.")
    return False


# MAIN PIPELINE
async def process_and_ingest():
    logger.info(f"Scanning knowledge base at: {DATA_DIR}")

    md_files = glob.glob(f"{DATA_DIR}/**/*.md", recursive=True)
    if not md_files:
        logger.warning("No .md files found. Add files to data/knowledge_base/ and retry.")
        return

    all_chunks: list[str] = []
    all_metadata: list[dict] = []

    # READ & CHUNK
    for file_path in md_files:
        filename = os.path.basename(file_path)
        # Use the immediate parent folder as category; fall back to "general"
        parent = os.path.basename(os.path.dirname(file_path))
        category = parent if parent != "knowledge_base" else "general"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Skipping '{filename}': could not read file — {e}")
            continue

        if not content.strip():
            logger.warning(f"Skipping '{filename}': file is empty.")
            continue

        chunks = chunk_by_headings(content)
        logger.info(f"'{filename}' → {len(chunks)} chunks (category: {category})")

        for local_idx, chunk in enumerate(chunks):
            vector_id = stable_vector_id(filename, local_idx, chunk)
            all_chunks.append(chunk)
            all_metadata.append({
                "id": vector_id,          # store for record assembly below
                "text": chunk,
                "source": filename,
                "category": category,
                "chunk_index": local_idx,
            })

    if not all_chunks:
        logger.warning("All .md files were empty or unreadable. Nothing to ingest.")
        return

    # EMBED
    logger.info(f"Requesting embeddings for {len(all_chunks)} chunks via OpenAI…")
    try:
        vectors = await get_embeddings_batch(all_chunks)
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        return

    if not vectors or len(vectors) != len(all_chunks):
        logger.error("Embedding count mismatch. Aborting.")
        return

    # UPSERT TO PINECONE
    records = [
        (meta["id"], vector, {k: v for k, v in meta.items() if k != "id"})
        for meta, vector in zip(all_metadata, vectors)
    ]

    upsert_tasks = [
        upsert_with_retry(records[i: i + BATCH_SIZE], namespace="career-portfolio")
        for i in range(0, len(records), BATCH_SIZE)
    ]

    logger.info(f"Upserting {len(upsert_tasks)} batch(es) in parallel…")
    results = await asyncio.gather(*upsert_tasks)

    success = sum(results)
    total = len(results)

    if success == total:
        logger.info(f"All {total} batch(es) upserted successfully!")
    else:
        logger.warning(f"{success}/{total} batch(es) succeeded. Check logs above for failures.")


if __name__ == "__main__":
    asyncio.run(process_and_ingest())