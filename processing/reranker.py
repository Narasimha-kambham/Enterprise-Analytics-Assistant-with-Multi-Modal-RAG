from sentence_transformers import CrossEncoder
from config.settings import RERANKER_MODEL, TOP_K_RERANK

# Load once globally (important for performance)
cross_encoder = CrossEncoder(RERANKER_MODEL)

def rerank(query, documents, top_k=TOP_K_RERANK):
    """
    Enterprise-level cross-encoder reranking.
    Takes retrieved docs and reorders by semantic relevance.
    """

    # Prepare pairs: (query, document_text)
    pairs = [(query, doc.page_content) for doc in documents]

    # Get relevance scores
    scores = cross_encoder.predict(pairs)

    # Attach scores to documents
    scored_docs = list(zip(documents, scores))

    # Sort by score (descending)
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Return top_k documents only
    return [doc for doc, score in scored_docs[:top_k]]