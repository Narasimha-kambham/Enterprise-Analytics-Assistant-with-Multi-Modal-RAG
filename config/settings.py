from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# --- Base Directories ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PDF_DIR = DATA_DIR / "pdfs"
CSV_DIR = DATA_DIR / "csv"
TEXT_DIR = DATA_DIR / "text"
IMAGE_DIR = DATA_DIR / "images"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"

# --- Persistence ---
VECTOR_STORE_PATH = BASE_DIR / "vector_store"
VISION_CACHE_PATH = BASE_DIR / "vision_cache.json"

# --- External Engine Paths (Windows Example) ---
TESSERACT_PATH = r"D:\program files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"D:\program files\poppler-25.12.0\Library\bin"

# --- Model Configurations ---
GEMINI_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Processing Parameters ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5
TEXT_DENSITY_THRESHOLD = 100
