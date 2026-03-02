from ingestion.router import ingest_all
from processing.chunking import chunk_documents
from processing.embeddings import build_or_load_vector_store
from processing.reranker import rerank
from llm.generator import generate_structured_answer


def initialize_system():
    print("\n" + "=" * 72)
    print("ENTERPRISE MULTI-MODAL ANALYTICS PLATFORM")
    print("Retrieval-Augmented Financial Intelligence System")
    print("=" * 72)

    print("\n[1/3] Data Ingestion Phase")
    documents = ingest_all("data")
    print(f"      Documents successfully ingested: {len(documents)}")

    print("\n[2/3] Document Structuring Phase")
    chunks = chunk_documents(documents)
    print(f"      Text chunks generated for retrieval: {len(chunks)}")

    print("\n[3/3] Vector Index Initialization")
    vector_store = build_or_load_vector_store(chunks)
    print("      Vector index ready for semantic retrieval.")

    print("\nSystem initialization complete.")
    print("The platform is now ready to process analytical queries.")
    print("Type 'exit' to terminate the session.\n")

    return vector_store


from config.settings import TOP_K_RETRIEVAL, TOP_K_RERANK

def handle_query(vector_store, query):

    print("\n--- Query Processing Pipeline Initiated ---")

    # Stage 1: Dense Retrieval
    print("Stage 1: Dense Vector Retrieval (FAISS)")
    retrieved_docs = vector_store.similarity_search(query, k=TOP_K_RETRIEVAL)
    print(f"         Candidate documents retrieved: {len(retrieved_docs)}")

    # Stage 2: Cross-Encoder Reranking
    print("Stage 2: Cross-Encoder Semantic Reranking")
    final_docs = rerank(query, retrieved_docs, top_k=TOP_K_RERANK)
    print("         Top 5 most relevant sources selected.")

    # Stage 3: LLM Synthesis
    print("Stage 3: Structured Response Generation (LLM)")
    answer = generate_structured_answer(query, final_docs)

    print("\n" + "=" * 72)
    print("ANALYTICAL RESPONSE")
    print("=" * 72 + "\n")
    print(answer)
    print("\n" + "=" * 72)
    print("End of Response\n")


def main():
    vector_store = initialize_system()

    while True:
        query = input("Enter analytical query: ").strip()

        if query.lower() in ["exit", "quit"]:
            print("\nSession terminated. Thank you.\n")
            break

        if not query:
            print("Please enter a valid query.")
            continue

        handle_query(vector_store, query)


if __name__ == "__main__":
    main()