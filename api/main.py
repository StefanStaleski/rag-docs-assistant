from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../retrieval"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../mlflow"))

from retriever import retrieve
from generator import generate
from tracker import track_query

app = FastAPI(title="RAG Docs Assistant")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class QueryRequest(BaseModel):
    question: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    start = time.time()
    chunks = retrieve(request.question, k=request.k)
    result = generate(request.question, chunks, GEMINI_API_KEY)
    latency_ms = (time.time() - start) * 1000

    track_query(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        chunks=chunks,
        latency_ms=latency_ms
    )

    return QueryResponse(answer=result["answer"], sources=result["sources"])