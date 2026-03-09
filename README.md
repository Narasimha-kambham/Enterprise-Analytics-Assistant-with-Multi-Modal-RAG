# Enterprise Multi-Modal Retrieval-Augmented Generation (RAG) System

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![LangChain](https://img.shields.io/badge/LangChain-RAG%20Pipeline-orange) ![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-green) ![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash%20Lite-purple) ![Architecture](https://img.shields.io/badge/Architecture-Multi--Modal%20RAG-red)

## 1. Project Overview

This repository contains a modular **Multi-Modal Retrieval-Augmented Generation (RAG)** system designed for financial analytics and document intelligence. The system ingests, processes, and semantically retrieves information across multiple data formats, including:

* Native and scanned PDFs
* Structured CSV datasets
* Standalone images (charts and infographics)
* Plain text documents

A two-stage retrieval pipeline — combining dense vector search with cross-encoder reranking — ensures that highly relevant context is provided to the Large Language Model (LLM) for accurate, grounded, and citation-aware responses.

This implementation emphasizes robustness, configurability, and practical enterprise constraints such as OCR fallback, caching, and persistent indexing.

---

## System Demo

This video demonstrates the full workflow of the Enterprise Multi-Modal RAG system.

<p align="center">
  <a href="https://youtu.be/j5cWtbepxfc">
    <img src="https://img.youtube.com/vi/j5cWtbepxfc/maxresdefault.jpg" width="600">
  </a>
</p>

<p align="center">
  ▶ Click the thumbnail to watch the full demo
</p>

---

## 2. Alignment with Evaluation Requirements

This implementation satisfies all functional and advanced requirements outlined in the evaluation brief:

### Functional Requirements

✔ Multi-Datatype Support
✔ Unified Embedding & Indexing Pipeline
✔ Vector Database Integration (FAISS)
✔ Semantic Top-k Retrieval
✔ Intelligent Query Interface

### Advanced Requirements Implemented

✔ Cross-Encoder Reranking
✔ Vision Result Caching
✔ Metadata-Aware Retrieval
✔ Config-Driven System Design

---

## 3. System Architecture

The system follows a modular pipeline architecture separating ingestion, processing, retrieval, and generation.

```text
[ DATA INGESTION ]                     [ QUERY PIPELINE ]
        |                                      |
+-------+-------+                      +-------+-------+
|  PDF  |  CSV  |                      | User Question |
| (OCR) | (DFs) |                      +-------+-------+
|  IMG  |  TXT  |                              |
+-------+-------+                    [ DENSE RETRIEVAL ]
        |                            (FAISS + MiniLM-L6)
[ NORMALIZATION ]                              |
(LangChain Documents)                [ SEMANTIC RERANKING ]
        |                            (MS-MARCO Cross-Encoder)
[ CHUNKING FLOW ]                              |
(Recursive Split)                    [ STRUCTURED GENERATION ]
        |                            (Gemini 2.5 Flash Lite)
[ EMBEDDING OPS ]                              |
(Sentence-Transformers)             [ CITED ANALYTIC RESPONSE ]
        |
[ PERSISTENT VECTOR STORE ]
(Local FAISS Index)
```

---

## 4. End-to-End Data Flow

1. **Ingestion**
   Files are loaded from paths defined in `config/settings.py`. Each datatype is routed to a dedicated extractor.

2. **Normalization**
   Extracted content is converted into `LangChain Document` objects with metadata:

   * Source type
   * File name
   * Page number
   * Extraction method

3. **Chunking**
   Documents are split using `RecursiveCharacterTextSplitter`.

4. **Embedding**
   Chunks are embedded using `all-MiniLM-L6-v2`.

5. **Indexing**
   Embeddings are stored in a persistent FAISS index.

6. **Retrieval & Generation**

   * Stage 1: Dense retrieval (Top 20)
   * Stage 2: Cross-encoder reranking (Top 5)
   * Final structured answer generation via Gemini.

---

## 5. Supported Data Types

### PDF Documents

* Native text extraction via PyMuPDF (fitz)
* Heuristic scanned detection
* OCR fallback (Tesseract + OpenCV preprocessing)
* Table extraction (Tabula, JVM-backed)
* Embedded image extraction + vision description

### Structured CSV

* Financial datasets converted into semantically descriptive text blocks
* Supports stock prices, company metadata, and financial indicators

### Standalone Images

* Vision-based structured description via Gemini
* Indexed alongside textual content

### Plain Text

* UTF-8 text ingestion

---

## 6. Chunking Strategy

Chunking parameters are centralized in `config/settings.py`.

* `CHUNK_SIZE = 800`
* `CHUNK_OVERLAP = 150`

### Rationale

* 800 characters balance semantic completeness and embedding efficiency.
* 150-character overlap prevents loss of contextual continuity across boundaries.
* Parameters are configurable to adapt to domain-specific requirements.

---

## 7. Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (configured in `EMBEDDING_MODEL`)

### Justification

* 384-dimensional embeddings
* Efficient CPU execution
* Strong semantic recall performance
* No external embedding API dependency
* Suitable for local indexing in cost-sensitive environments

---

## 8. Vector Store Design (FAISS)

* Local persistent FAISS index stored in `VECTOR_STORE_PATH`
* Automatically loaded on restart
* Optimized for similarity search over normalized transformer embeddings

This ensures:

* Fast startup after first build
* No re-embedding required on every execution

---

## 9. Retrieval Strategy

A two-stage retrieval approach improves precision:

### Stage 1: Dense Retrieval

* FAISS returns Top 20 candidates (`TOP_K_RETRIEVAL`)

### Stage 2: Cross-Encoder Reranking

* Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (configured in `RERANKER_MODEL`)
* Evaluates query–document pairs jointly
* Top 5 (`TOP_K_RERANK`) passed to LLM

This reduces irrelevant context and improves answer grounding.

---

## 10. Advanced Features

### Centralized Configuration

All key parameters (paths, thresholds, model names) are defined in `config/settings.py`.

### Neural Reranking

Improves semantic precision beyond standard vector similarity.

### Vision Caching

Image descriptions are stored in `vision_cache.json` to:

* Avoid repeated API calls
* Reduce cost
* Improve latency

### OCR Fallback Strategy

Heuristic page sampling triggers OCR only when necessary (based on `TEXT_DENSITY_THRESHOLD`).

---

## 11. Vision & OCR Logic

### OCR Pipeline

* PDF page → Image conversion (Poppler)
* Grayscale + Otsu thresholding (OpenCV)
* Text extraction via Tesseract
* Configurable `TEXT_DENSITY_THRESHOLD`

### Vision Integration

* Extracted images converted to base64
* Sent to Gemini
* Structured description returned
* Indexed as text for unified retrieval

---

## 12. Caching Strategy

* **Vision Cache:** Stored at `VISION_CACHE_PATH`
* **Vector Store Cache:** Loaded from `VECTOR_STORE_PATH` if available

This prevents unnecessary recomputation and ensures stable repeated runs.

---

## 13. Tradeoffs & Design Decisions

| Decision                        | Rationale                         |
| ------------------------------- | --------------------------------- |
| Local embeddings (MiniLM)       | Avoid external embedding API cost |
| FAISS (CPU)                     | Works without GPU infrastructure  |
| Cross-Encoder reranking         | Improves retrieval precision      |
| Tabula (Java dependency)        | Better table extraction accuracy  |
| Config-based paths/constants    | Environment portability           |

---

## 14. Dependencies & System Requirements

### Python Dependencies

* langchain
* langchain-community
* langchain-huggingface
* langchain-google-genai
* sentence-transformers
* faiss-cpu
* pandas
* numpy
* pymupdf
* pytesseract
* pdf2image
* opencv-python
* tabula-py
* jpype1
* python-dotenv

### System Requirements

* Java JDK 17+
* Tesseract OCR
* Poppler (PDF rendering)

---

## 15. Installation

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

Update paths and model configurations in:

```
config/settings.py
```

---

## 16. Run

```bash
python main.py
```

System will:

1. Ingest data
2. Build or load FAISS index
3. Enter interactive query loop

Type `exit` to terminate.

---

## 17. Folder Structure

```text
.
├── config/             # Centralized settings and constants
├── data/               # Source data (PDF, CSV, Images, Text)
├── ingestion/          # Multi-modal extraction logic
├── llm/                # Gemini client and prompt engineering
├── processing/         # Chunking, Embeddings, and Reranking
├── vector_store/       # Persistent FAISS index files
├── main.py             # System entry point
├── vision_cache.json   # Cached vision API results
└── requirements.txt    # Python dependencies
```

---

## 18. Data Sources & Provenance

The system was validated using publicly available financial materials.

### Unstructured Narrative

* Microsoft 2024 Annual Report (SEC Form 10-K)
* Source: Microsoft Investor Relations Portal

### Structured Data

* S&P 500 historical dataset (Kaggle)

### Visual Context

* Publicly available investor charts and infographics

All materials were used strictly for system validation and demonstration.
The system does not redistribute proprietary content.

---

## 19. Why This Design is Enterprise-Ready

This system emphasizes:

* **Deterministic Ingestion:** Reliable extraction across heterogeneous formats.
* **Config-Driven Execution:** Decoupled business logic from environmental paths.
* **Citation-Aware Response:** Transparency in how the model reaches conclusions.
* **Precision-Focused Retrieval:** Two-stage strategy minimizes noise.
* **Cost-Aware Architecture:** Local embeddings and strategic result caching.

It demonstrates how multi-modal enterprise data can be unified into a consistent semantic retrieval pipeline while maintaining robustness against noisy inputs (scanned PDFs, charts, tables).


