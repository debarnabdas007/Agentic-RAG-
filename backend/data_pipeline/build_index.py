import os
import pickle
import fitz  # PyMuPDF >> PyPDF ... I am saying !!!
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

# Text claening
def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')  # removes excessive newlines, spaces,.. thus better for embediing quality later on
    return ' '.join(text.split())


# Chunking
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """ Custom sliding-window chunker """
    
    ## Infinite Loop Avoidance 
    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly less than chunk_size to prevent infinite loops !!")
        
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len: # Continue chunking until entire document processed..
        end = start + chunk_size # Creating chunk boundary
        
        # If we are not at the end of the text, snap back to the nearest space --> This avoids cutting mid-word !!
        if end < text_len:
            while end > start and text[end] != ' ':
                end -= 1
            # If no space was found (e.g., a giant string of code), force the cut
            if end == start:
                end = start + chunk_size
                
        chunk = text[start:end].strip() #removes extra spaces while extracting chunk
        if chunk:
            chunks.append(chunk) # Avoids empty chunks
            
        # SLIDING WINDOW :- Move start forward, accounting for overlap
        start = end - overlap
        

    return chunks

# Embedding + FAISS index
def build_vector_database():
    """ Read PDFs --> chunks --> embeds --> saves the FAISS index """
    
    project_root = os.getcwd()
    pdf_dir = os.path.abspath(os.path.join(project_root, settings.RAW_PDF_DIR))
    vector_dir = os.path.abspath(os.path.join(project_root, settings.VECTOR_STORE_DIR))
    
    os.makedirs(vector_dir, exist_ok=True)
    
    logger.info("Loading SentenceTransformer model (this may take a minute on first run)...")
    
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
    
    documents = []  # this goes to embeddings only..
    metadata = []  # but need to save the text and source:metadata, so the Agent can read it later.. used during retireval !!

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')] ## collecting ALL pdf file names
    if not pdf_files:
        logger.error(f"No PDFs found in {pdf_dir}. Run ingest.py first!")
        return

    logger.info(f"Found {len(pdf_files)} PDFs. Starting extraction and chunking...")

    # Extract and Chunk
    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        logger.info(f"Processing: {filename}")
        
        try:
            doc = fitz.open(filepath)
            for page_num, page in enumerate(doc): #Page by Page processing 
                text = page.get_text("text")        # Text extract
                clean_txt = clean_text(text)        # cleaning text
                
                if not clean_txt:                   # skip empty pages.. 
                    continue
                
                chunks = chunk_text(clean_txt, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP) ## Chunking.. defined before
                
                for chunk in chunks:
                    documents.append(chunk)         # store RAW chunk
                    metadata.append({               # store metadata
                        "source": filename,
                        "page": page_num + 1,
                        "text": chunk
                    })

            doc.close()

        except Exception as e:                      # This prevents one corrupted PDF crashing entire pipeline
            logger.error(f"Failed to process {filename}: {e}")

    logger.info(f"Total chunks created: {len(documents)}")

    # Embedding
    logger.info("Converting chunks into vector embeddings...")
    
    embeddings = model.encode(documents, batch_size=32, show_progress_bar=True, normalize_embeddings=True)  ## Normalization: vectors --> unit vector.. we need for cosine similarity.. as dot.product
    embeddings = np.array(embeddings).astype('float32') ## Generate embeddings as a np.array

    # Build FAISS Index
    logger.info("Building FAISS index...")
    embedding_dimension = embeddings.shape[1]
    """IndexFlatIP (Inner Product) on normalized vectors = Exact Cosine Similarity """
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings) # store all vectors inside FAISS

    # Saving..
    faiss_path = os.path.join(vector_dir, "index.faiss")
    meta_path = os.path.join(vector_dir, "metadata.pkl")

    faiss.write_index(index, faiss_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(f" Vector database successfully saved to {vector_dir}")
    logger.info(f"FAISS Index size: {index.ntotal} vectors.")

if __name__ == "__main__":
    build_vector_database()



## THis is the one-tiime setup for offline work.. separated it from the Fast Online Retrieval

'''
PDFs
  ↓
Text Extraction
  ↓
Cleaning
  ↓
Chunking
  ↓
Embeddings
  ↓
FAISS Index
  ↓
Metadata Persistence

'''   
## NOTE:-
'''
For retrieval:

FAISS returns vector index
↓
metadata[index]
↓
recover actual chunk text

'''