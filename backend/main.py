from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.rag_pipeline import run_system_a, run_system_b, run_system_c

app = FastAPI()

# CORS is required so our frontend can securely talk to our backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected JSON format from the frontend
class QueryRequest(BaseModel):
    query: str

@app.post("/api/ask")
def ask_systems(request: QueryRequest):
    """
    Receives a single question, processes it through A, B, and C, 
    and returns all three results mapped to JSON.
    """
    question = request.query
    
    # Execute the systems (updated System C call for Stage 2 pipeline)
    ans_a = run_system_a(question)
    ans_b, docs_b = run_system_b(question, k=2)
    ans_c, docs_c = run_system_c(question, k=5)
    
    return {
        "system_a": {
            "answer": ans_a
        },
        "system_b": {
            "answer": ans_b,
            "retrieved_chunks": docs_b
        },
        "system_c": {
            "answer": ans_c,
            "filtered_chunks": [{"text": t, "score": s} for t, s in docs_c]
        }
    }