import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Initialize global resources once
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Ensure it finds the data folder relative to this file
base_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_dir, "data", "prototype_qa_data.csv")

df = pd.read_csv(data_path)
knowledge_base = df['knowledge'].dropna().tolist()

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
document_embeddings = embedder.encode(knowledge_base, convert_to_numpy=True, normalize_embeddings=True)

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)

# 2. System A
def run_system_a(query):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
            {"role": "user", "content": query}
        ],
        temperature=0.0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# 3. System B
def run_system_b(query, k=2):
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(query_vec, k)
    
    retrieved_docs = [knowledge_base[i] for i in indices[0] if i != -1]
    context_str = "\n".join(retrieved_docs)
    
    prompt = f"Answer the question based strictly on the context.\n\nContext:\n{context_str}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip(), retrieved_docs

# 4. System C
def run_system_c(query, k=3, threshold=0.45):
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(query_vec, k)
    
    filtered_docs = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score >= threshold:
            filtered_docs.append((knowledge_base[idx], float(score))) # Cast numpy float to standard float for JSON
            
    if not filtered_docs:
        context_str = "No sufficiently relevant context available."
    else:
        context_str = "\n".join([doc for doc, _ in filtered_docs])
        
    prompt = f"Answer the question based strictly on the context.\n\nContext:\n{context_str}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )
    return response.choices[0].message.content.strip(), filtered_docs