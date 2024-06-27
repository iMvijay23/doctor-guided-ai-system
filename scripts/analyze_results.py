import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer

def score_text(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits.squeeze().cpu().tolist()
    return scores[0] if isinstance(scores, list) else scores

def compare_claims(original_claims, empathy_claims, sentence_model, similarity_threshold=0.9):
    original_embeddings = sentence_model.encode(original_claims)
    empathy_embeddings = sentence_model.encode(empathy_claims)
    
    similarity_matrix = []  # Set similarity_matrix as empty

    matched_claims = []
    for i, empathy_claim in enumerate(empathy_claims):
        max_similarity = 0  # Set to 0 or any default value since no similarity is calculated
        if max_similarity >= similarity_threshold:
            original_claim_index = 0  # Default index since no calculation is done
            matched_claims.append({
                'empathy_claim': empathy_claim,
                'original_claim': original_claims[original_claim_index],
                'similarity': max_similarity
            })
    
    return matched_claims, similarity_matrix

def calculate_metrics(original_claims, empathy_claims, matched_claims):
    true_positives = len(matched_claims)
    false_positives = len(empathy_claims) - true_positives
    false_negatives = len(original_claims) - true_positives
    
    precision = true_positives / len(empathy_claims) if len(empathy_claims) > 0 else 0
    recall = true_positives / len(original_claims) if len(original_claims) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def visualize_scores(df, score_type):
    plt.figure(figsize=(10, 6))
    plt.boxplot([df[f'original_{score_type}'], df[f'empathy_{score_type}']])
    plt.title(f'{score_type.capitalize()} Scores')
    plt.xticks([1, 2], ['Original', 'Empathy Injected'])
    plt.ylabel('Score')
    plt.savefig(f"{score_type.lower()}_scores2.png")
    plt.close()

def process_results(results_file, empathy_model_name, quality_model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    empathy_model, empathy_tokenizer = load_model(empathy_model_name, device)
    quality_model, quality_tokenizer = load_model(quality_model_name, device)
    
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    results = []

    with open(results_file, 'r') as f:
        for line in tqdm(f):
            result = json.loads(line)
            
            primary_id = result['primaryid']
            original_response = result['original_response']
            empathy_response = result['empathetic_response']
            original_claims = result['key_facts'].split('\n')
            empathy_claims = list(result['supported_facts'].keys())

            # Calculate scores
            orig_empathy = score_text(empathy_model, empathy_tokenizer, original_response, device)
            emp_empathy = score_text(empathy_model, empathy_tokenizer, empathy_response, device)
            orig_quality = score_text(quality_model, quality_tokenizer, original_response, device)
            emp_quality = score_text(quality_model, quality_tokenizer, empathy_response, device)
            
            # Compare claims
            matched_claims, similarity_matrix = compare_claims(original_claims, empathy_claims, sentence_model)
            
            # Calculate metrics
            precision, recall, f1 = calculate_metrics(original_claims, empathy_claims, matched_claims)

            results.append({
                'primary_id': primary_id,
                'original_empathy': orig_empathy,
                'empathy_empathy': emp_empathy,
                'original_quality': orig_quality,
                'empathy_quality': emp_quality,
                'original_claim_count': len(original_claims),
                'empathy_claim_count': len(empathy_claims),
                'matched_claim_count': len(matched_claims),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'matched_claims': matched_claims,
                'similarity_matrix': similarity_matrix  # This will be an empty list
            })

    df = pd.DataFrame(results)

    # Perform statistical tests
    empathy_ttest = stats.ttest_rel(df['original_empathy'], df['empathy_empathy'])
    quality_ttest = stats.ttest_rel(df['original_quality'], df['empathy_quality'])
    
    # Visualize results
    visualize_scores(df, 'empathy')
    visualize_scores(df, 'quality')
    
    # Save detailed results
    df.to_json('detailed_results_2.json', orient='records', lines=True)
    
    return empathy_ttest, quality_ttest, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="Path to the results file")
    parser.add_argument("--empathy_model_name", type=str, required=True, help="Empathy model name on Hugging Face")
    parser.add_argument("--quality_model_name", type=str, required=True, help="Quality model name on Hugging Face")

    args = parser.parse_args()

    empathy_ttest, quality_ttest, df = process_results(args.results_file, args.empathy_model_name, args.quality_model_name)

    print(f"Empathy t-test: {empathy_ttest}")
    print(f"Quality t-test: {quality_ttest}")
    print(f"Average original claim count: {df['original_claim_count'].mean()}")
    print(f"Average empathy claim count: {df['empathy_claim_count'].mean()}")
    print(f"Average matched claim count: {df['matched_claim_count'].mean()}")
    print(f"Average precision: {df['precision'].mean()}")
    print(f"Average recall: {df['recall'].mean()}")
    print(f"Average F1 score: {df['f1'].mean()}")
    print(f"Average original quality score: {df['original_quality'].mean()}")
    print(f"Average empathy quality score: {df['empathy_quality'].mean()}")
