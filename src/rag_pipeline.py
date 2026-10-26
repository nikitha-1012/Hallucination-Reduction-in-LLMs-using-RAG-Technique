import os
import time
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. Initialize global resources
client = Groq(api_key="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")

# Use the globally supported model for all API calls
MODEL_NAME = "openai/gpt-oss-20b"

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

# 2. System A (Baseline LLM)
def run_system_a(query):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# 3. System B (Naive RAG)
def run_system_b(query, k=2):
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, indices = index.search(query_vec, k)
    
    retrieved_docs = [knowledge_base[i] for i in indices[0] if i != -1]
    context_str = "\n".join(retrieved_docs)
    
    prompt = f"Answer the question based strictly on the context.\n\nContext:\n{context_str}\n\nQuestion: {query}"
    try:
        response = client.chat.completions.create(
           model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip(), retrieved_docs
    except Exception as e:
        return f"Error: {e}", []

# 4. Phase 2: AI Evaluators (Judge & Verifier)
def judge_chunk_relevance(query, chunk_text):
    """Filters input context before generation."""
    prompt = (
        f"You are a strict relevance judge. Read the user's question and the retrieved context.\n"
        f"Does the context contain factual evidence that helps answer the question?\n"
        f"You must reply with exactly one word: YES or NO.\n\n"
        f"Question: {query}\n"
        f"Context: {chunk_text}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return "YES" in response.choices[0].message.content.strip().upper()
    except:
        return False

def verify_answer(query, context, generated_answer):
    """Verifies output consistency after generation."""
    prompt = (
        f"You are a strict fact-checker. Read the CONTEXT, the QUESTION, and the GENERATED ANSWER.\n"
        f"Does the CONTEXT fully support the claims made in the GENERATED ANSWER?\n"
        f"Reply with exactly one word: SUPPORTED or UNSUPPORTED.\n\n"
        f"CONTEXT: {context}\n"
        f"QUESTION: {query}\n"
        f"GENERATED ANSWER: {generated_answer}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return "SUPPORTED" in response.choices[0].message.content.strip().upper()
    except:
        return False

def remove_redundant_chunks(filtered_docs, similarity_cutoff=0.85):
    """Removes text chunks that are near-duplicates of already selected chunks."""
    unique_docs = []
    for doc, score in filtered_docs:
        is_redundant = False
        for u_doc, _ in unique_docs:
            words_doc = set(doc.lower().split())
            words_u = set(u_doc.lower().split())
            if len(words_doc) > 0 and len(words_u) > 0:
                intersection = len(words_doc.intersection(words_u))
                union = len(words_doc.union(words_u))
                jaccard = intersection / union if union > 0 else 0
                if jaccard > similarity_cutoff:
                    is_redundant = True
                    break
        if not is_redundant:
            unique_docs.append((doc, score))
    return unique_docs

# 5. System C (Advanced Semantic Filtering, Redundancy Removal, & Fallback Loop)
def run_system_c(query, k=5):
    """Advanced System C with Redundancy Filtering, Verification, and Fallback Loop."""
    
    def execute_rag_attempt(active_k):
        # Step 1: Retrieval
        query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = index.search(query_vec, active_k)
        
        # Step 2: Dynamic Semantic Evidence Filtering
        filtered_docs = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                chunk_text = knowledge_base[idx]
                if judge_chunk_relevance(query, chunk_text):
                    filtered_docs.append((chunk_text, float(score)))
                time.sleep(0.3)
                
        # Step 3: Redundancy Filtering
        unique_docs = remove_redundant_chunks(filtered_docs)
        
        if not unique_docs:
            return None, []
            
        context_str = "\n\n".join([f"[{i+1}] {doc}" for i, (doc, _) in enumerate(unique_docs)])
        
        # Step 4: Generation
        prompt = f"Answer the question strictly using the provided context.\n\nContext:\n{context_str}\n\nQuestion: {query}"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150
            )
            answer = response.choices[0].message.content.strip()
            
            # Step 5: Post-Generation Consistency Check
            time.sleep(0.3)
            if not verify_answer(query, context_str, answer):
                return None, unique_docs # Triggers fallback
                
            return answer, unique_docs
        except Exception:
            return None, unique_docs

    # --- Attempt 1: Standard Search (K=5) ---
    answer, docs = execute_rag_attempt(k)
    
    # --- Attempt 2: Fallback Loop (Widening Net to K=8 if Attempt 1 fails) ---
    if not answer:
        print("[System C Fallback Triggered]: Broadening retrieval search space...")
        answer, docs = execute_rag_attempt(active_k=8)
        if not answer:
            answer = "[FALLBACK EXHAUSTED]: Could not verify a reliable supported answer after re-retrieval."
            
    return answer, docs