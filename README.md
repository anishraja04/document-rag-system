# Intelligent Document RAG & QA System

A Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, LangChain, and FAISS. It processes text documents, creates semantic chunks, generates vector embeddings using Sentence Transformers, and retrieves context to answer user queries.

## Features
- **Document Ingestion API:** Upload documents, split them into semantic chunks, and generate vector embeddings.
- **Vector Search:** Uses FAISS (Facebook AI Similarity Search) for blazing-fast similarity retrieval.
- **Embeddings:** Uses HuggingFace `all-MiniLM-L6-v2` for dense vector representations.
- **FastAPI:** Exposes clean REST endpoints for integration.

## Setup & Run Locally
1. Clone the repository.
2. Build and run using Docker:
   ```bash
   docker-compose up -d --build
   ```
3. Access Swagger UI at `http://localhost:8001/docs`.

## Endpoints
- `POST /ingest`: Upload a text document.
- `POST /ask`: Ask a question based on ingested context.


## Community
Contributions are always welcome. See CONTRIBUTING.md for details.
