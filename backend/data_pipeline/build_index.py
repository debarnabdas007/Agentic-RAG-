import pickle
from pathlib import Path

import faiss
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.paths import project_root
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Character-based sliding-window chunker (chunk_size/overlap are in characters)."""
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size.")

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end < text_len:
            while end > start and text[end] != " ":
                end -= 1
            if end == start:
                end = start + chunk_size

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def build_vector_database() -> None:
    """Read PDFs → chunk → embed → save FAISS index and metadata pickle."""
    root = project_root()
    pdf_dir = root / settings.RAW_PDF_DIR
    vector_dir = root / settings.VECTOR_STORE_DIR
    vector_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SentenceTransformer model (first run may download weights)...")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    documents: list[str] = []
    metadata: list[dict] = []

    pdf_files = sorted(p for p in pdf_dir.iterdir() if p.suffix.lower() == ".pdf")
    if not pdf_files:
        logger.error("No PDFs found in %s. Run ingest first.", pdf_dir)
        return

    logger.info("Found %s PDFs. Starting extraction and chunking...", len(pdf_files))

    for filepath in pdf_files:
        logger.info("Processing: %s", filepath.name)
        try:
            doc = fitz.open(filepath)
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                clean_txt = clean_text(text)
                if not clean_txt:
                    continue
                chunks = chunk_text(
                    clean_txt, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
                )
                for chunk in chunks:
                    documents.append(chunk)
                    metadata.append(
                        {
                            "source": filepath.name,
                            "page": page_num + 1,
                            "text": chunk,
                        }
                    )
            doc.close()
        except Exception as e:
            logger.error("Failed to process %s: %s", filepath.name, e)

    logger.info("Total chunks created: %s", len(documents))
    if not documents:
        logger.error("No text extracted; aborting index build.")
        return

    logger.info("Encoding chunks...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings).astype("float32")

    logger.info("Building FAISS index (IndexFlatIP = cosine similarity on L2-normalized vectors)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = vector_dir / "index.faiss"
    meta_path = vector_dir / "metadata.pkl"
    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info("Vector database saved under %s", vector_dir)
    logger.info("FAISS index size: %s vectors.", index.ntotal)


if __name__ == "__main__":
    build_vector_database()
