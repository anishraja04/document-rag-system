from fastapi import FastAPI, UploadFile, File, HTTPException
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
