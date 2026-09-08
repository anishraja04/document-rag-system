import os

files = {
    'requirements.txt': '''fastapi==0.103.1
uvicorn==0.23.2
langchain==0.0.316
faiss-cpu==1.7.4
sentence-transformers==2.2.2
pydantic==2.3.0
python-multipart==0.0.6
''',
    'app/__init__.py': '',
    'app/main.py': '''from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.rag import ingest_document, answer_question
import shutil
import os

app = FastAPI(title="Intelligent Document RAG System")

class Query(BaseModel):
    question: str

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = ingest_document(file_location)
        os.remove(file_location)
        return {"message": "Document ingested successfully", "chunks": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(query: Query):
    try:
        answer = answer_question(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
''',
    'app/rag.py': '''from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import os

# Initialize embeddings (Sentence Transformers)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store_path = "faiss_index"

def get_vector_store():
    if os.path.exists(vector_store_path):
        return FAISS.load_local(vector_store_path, embeddings)
    return None

def ingest_document(file_path: str):
    # For simplicity, assuming text files
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Semantic chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Generate embeddings and store
    vector_store = get_vector_store()
    if vector_store is None:
        vector_store = FAISS.from_documents(texts, embeddings)
    else:
        vector_store.add_documents(texts)
    
    vector_store.save_local(vector_store_path)
    return len(texts)

def answer_question(question: str):
    vector_store = get_vector_store()
    if not vector_store:
        return "No documents ingested yet. Please upload a document first."
    
    # Retrieve relevant context
    docs = vector_store.similarity_search(question, k=3)
    context = "\\n".join([doc.page_content for doc in docs])
    
    # In a real app, we would pass this context to an LLM (OpenAI, local Llama, etc.)
    # Mocking LLM response generation based on context for demonstration
    if not context:
        return "I could not find relevant context to answer your question."
        
    return f"Based on the provided documents, here is the relevant context:\\n{context}\\n\\n(LLM generation would synthesize this context into a final answer)."

''',
    'Dockerfile': '''FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
''',
    'docker-compose.yml': '''version: '3.8'
services:
  rag_api:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./faiss_index:/app/faiss_index
    restart: always
''',
    '.github/workflows/deploy.yml': '''name: Deploy RAG System

on:
  workflow_dispatch:
  push:
    branches:
      - master
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v3

      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            if [ ! -d "document-rag-system" ]; then
              git clone https://github.com/${{ github.repository }}.git document-rag-system
            fi
            cd document-rag-system
            git pull origin master
            
            sudo docker-compose down
            sudo docker-compose up -d --build
''',
    '.gitignore': '''__pycache__/
*.pyc
.env
venv/
faiss_index/
temp_*
''',
    'README.md': '''# Intelligent Document RAG & QA System

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
'''
}

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Files created.")
