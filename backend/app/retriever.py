import pickle
import numpy as np
import faiss
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from backend.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class AdvancedRetriever:

    def __init__(self):
        logger.info("Initializing Advanced Retriever...")
        
        # Loading Semantic Embedding Model
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Loading Cross-Encoder Reranker
        logger.info("Loading Cross-Encoder Reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Loading FAISS and Metadata using `pathlib` (NOTE)
        project_root = Path(__file__).resolve().parent.parent.parent
        faiss_path = project_root / settings.VECTOR_STORE_DIR / "index.faiss"
        meta_path = project_root / settings.VECTOR_STORE_DIR / "metadata.pkl"
        
        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Vector index not found. Run build_index.py first.")
            
        self.index = faiss.read_index(str(faiss_path))  # loading FAISS index into RAM
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)              # loading Metadata
            
        # BM25 keyword search
        logger.info("Building BM25 Keyword Corpus...")
        tokenized_corpus = [re.findall(r"\w+", doc['text'].lower()) for doc in self.metadata] # tokenization
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        logger.info("Retriever initialized and ready.")



    def _semantic_search(self, query: str, k: int) -> dict:
        query_vector = self.embedder.encode([query], normalize_embeddings=True).astype('float32')  # NOTE: NORMALIZE_embedding : we r making unit vectors HERE, for cosine similarity later on !!
        distances, indices = self.index.search(query_vector, k)
        
        results = {} # Rank Dictionary !
        for rank, idx in enumerate(indices[0]):
            if idx != -1:
                results[idx] = rank
        return results

    def _keyword_search(self, query: str, k: int) -> dict:
        
        tokenized_query = re.findall(r"\w+", query.lower()) # Regrex Tokenization for query
        scores = self.bm25.get_scores(tokenized_query)      # keyword Relevance score for every chunk
        top_indices = np.argsort(scores)[::-1][:k]          # sorted by score.. but Descending
        
        results = {}
        for rank, idx in enumerate(top_indices):
            results[idx] = rank
        return results



    def retrieve_and_rerank(
        self,
        query: str,
        *,
        use_reranker: bool = True,
        apply_score_threshold: bool | None = None,
    ) -> list[dict]:
        """
        Hybrid RRF retrieval, optional cross-encoder rerank + threshold filter.
        When use_reranker=False (ablations), results stay in RRF order without cross-encoder scores.
        """
        if not query or not query.strip():
            logger.warning("Empty query received. Returning empty retrieval.")
            return []
            
        logger.info(f"Executing retrieval for query: '{query}' (use_reranker={use_reranker})")
        
        semantic_ranks = self._semantic_search(query, settings.RETRIEVER_TOP_K)
        keyword_ranks = self._keyword_search(query, settings.RETRIEVER_TOP_K)
        
        # Observability traces (for Retrieval Transparency)
        logger.info(f"Top Semantic hit indices: {list(semantic_ranks.keys())[:3]}")
        logger.info(f"Top BM25 hit indices: {list(keyword_ranks.keys())[:3]}")
        

        # Reciprocal Rank Fusion (RRF) : use ranking positions to merge BM25 and cosine similarity
        rrf_scores = {}
        k_constant = 60 
        
        all_indices = set(semantic_ranks.keys()).union(set(keyword_ranks.keys()))
        for idx in all_indices:
            score = 0.0
            if idx in semantic_ranks:
                score += 1.0 / (k_constant + semantic_ranks[idx])
            if idx in keyword_ranks:
                score += 1.0 / (k_constant + keyword_ranks[idx])
            rrf_scores[idx] = score
            
        fused_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:settings.RETRIEVER_TOP_K]
        
        # Deep copy to prevent state mutation (NOTE)
        candidate_docs = [self.metadata[idx].copy() for idx in fused_indices]

        if use_reranker:
            sentence_pairs = [[query, doc['text']] for doc in candidate_docs]
            cross_scores = self.reranker.predict(sentence_pairs)
            logger.info(f"Top reranker scores: {cross_scores[:5]}")
            for i, doc in enumerate(candidate_docs):
                doc['rerank_score'] = float(cross_scores[i])
            candidate_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        else:
            for i, doc in enumerate(candidate_docs):
                idx = fused_indices[i]
                doc['rerank_score'] = float(rrf_scores.get(idx, 0.0))
            logger.info("Reranker skipped (ablation / diagnostic mode).")

        if apply_score_threshold is None:
            apply_score_threshold = use_reranker

        threshold = settings.RERANK_THRESHOLD
        if apply_score_threshold:
            final_results = [
                doc for doc in candidate_docs
                if doc['rerank_score'] > threshold
            ][:settings.RERANKER_TOP_K]
        else:
            final_results = candidate_docs[: settings.RERANKER_TOP_K]

        logger.info(
            f"Retrieved {len(final_results)} chunks "
            f"(threshold={threshold if apply_score_threshold else 'off'}, reranker={use_reranker})."
        )
        return final_results







'''
User Query
   ↓

Semantic Search (FAISS)
   +
Keyword Search (BM25)

   ↓
Reciprocal Rank Fusion (RRF)

   ↓
Cross-Encoder Reranking

   ↓
Threshold Filtering

   ↓
Top Final Chunks        

'''



"""
Why Retriever is Shared

Because retriever is READ-ONLY mostly.

It only:

searches FAISS
searches BM25
reranks

It does NOT store conversation data.

So sharing retriever is safe.

actually a good optimization !!

"""