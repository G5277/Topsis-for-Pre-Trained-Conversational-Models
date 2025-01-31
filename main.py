import torch
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd

# Load pre-trained models
models = {
    "DialoGPT": pipeline("text-generation", model="microsoft/DialoGPT-large"),
    "BERT": pipeline("fill-mask", model="bert-base-uncased"),
    "GPT-2": pipeline("text-generation", model="gpt2"),
    "T5": pipeline("text2text-generation", model="t5-small"),
   # "LLaMA-2": pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf"),
    "Falcon": pipeline("text-generation", model="tiiuae/falcon-7b"),
}

# Function to generate text
def generate_text(model, prompt):
    try:
        return model(prompt)
    except Exception as e:
        return f"Error: {str(e)}"

# Function to compute BLEU and ROUGE scores for generated text
def compute_metrics(generated_text, reference):
    # BLEU Score
    generated_tokens = generated_text.split()
    bleu_score = sentence_bleu(reference, generated_tokens) if generated_tokens else 0.0

    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(" ".join(reference[0]), generated_text) if generated_text else {
        "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0
    }

    return {
        "BLEU": round(bleu_score, 4),
        "ROUGE-1": round(rouge_scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(rouge_scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(rouge_scores["rougeL"].fmeasure, 4),
    }

# Sample input prompt
input_prompt = "How do large language models work?"
reference = [["Large", "language", "models", "use", "neural", "networks", "to", "process", "text"]]  # Reference tokens for BLEU

# Store results
results = {}

# Generate text and compute metrics for each model
for model_name, model in models.items():
    generated_text = generate_text(model, input_prompt)
    print(f"\n🔹 {model_name} Response: {generated_text}")

    # Ensure the generated text is in the correct format
    generated_text_str = generated_text[0]['generated_text'] if isinstance(generated_text, list) else generated_text

    # Compute and store actual metrics
    results[model_name] = compute_metrics(generated_text_str, reference)

# Display results as a table
df = pd.DataFrame.from_dict(results, orient="index")
print("\n📊 Model Performance Metrics:")
print(df)