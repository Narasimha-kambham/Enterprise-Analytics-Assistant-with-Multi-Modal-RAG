from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config.settings import EMBEDDING_MODEL, VECTOR_STORE_PATH
import os

def build_or_load_vector_store(chunks, path=VECTOR_STORE_PATH):

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    if os.path.exists(os.path.join(path, "index.faiss")):
        print("📦 Loading existing FAISS index...")
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    print("🧠 Creating FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(path)
    return vector_store