import os
import pandas as pd
from src.rag_pipeline import run_system_a, run_system_b, run_system_c

def evaluate_pipeline(csv_path="data/prototype_qa_data.csv", max_samples=5):
    print("Loading evaluation dataset...")
    if not os.path.exists(csv_path):
        # Fallback path check if run from a different directory
        csv_path = "../data/prototype_qa_data.csv"
        
    df = pd.read_csv(csv_path)
    eval_df = df.head(max_samples)
    
    results = []
    
    for idx, row in eval_df.iterrows():
        question = row['question']
        reference_answer = row['right_answer']
        
        print(f"\nProcessing [{idx+1}/{max_samples}]: {question}")
        
        # Run all three systems
        ans_a = run_system_a(question)
        ans_b, _ = run_system_b(question, k=2)
        ans_c, _ = run_system_c(question, k=5)
        
        results.append({
            "question": question,
            "reference": reference_answer,
            "system_a": ans_a,
            "system_b": ans_b,
            "system_c": ans_c
        })
        
    results_df = pd.DataFrame(results)
    
    # Simple Exact Match Calculation Function
    def calc_exact_match(preds, refs):
        matches = [1 if p.strip().lower() == r.strip().lower() else 0 for p, r in zip(preds, refs)]
        return sum(matches) / len(matches) if matches else 0
        
    print("\n" + "="*50)
    print("QUANTITATIVE EVALUATION RESULTS")
    print("="*50)
    
    refs = results_df['reference'].tolist()
    
    for sys_name in ['system_a', 'system_b', 'system_c']:
        preds = results_df[sys_name].tolist()
        em = calc_exact_match(preds, refs)
        
        print(f"\n--- {sys_name.upper()} ---")
        print(f"Exact Match Score: {em:.2f}")
        
    # Save results to the results folder
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/evaluation_summary.csv", index=False)
    print("\nEvaluation complete! Results saved to results/evaluation_summary.csv")

if __name__ == "__main__":
    evaluate_pipeline()