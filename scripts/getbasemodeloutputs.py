import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import argparse

def load_data(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)
    return data

def save_results(results, output_path):
    with open(output_path, 'w') as file:
        json.dump(results, file, indent=4)

def generate_responses(model, tokenizer, data, device, batch_size=1):
    results = []
    for item in tqdm(data):
        query = item['question']
        prompt = f"###Instruction: Please provide a comprehensive and detailed response to the following patient question.\n###Patient Query: {query}\n###BaseResponse:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract text after "###BaseResponse:"
        if "###BaseResponse:" in response:
            response = response.split("###BaseResponse:")[1].strip()
        
        results.append({
            'primaryid': item.get('primaryid', None),
            'question': query,
            'base_response': response
        })
    
    return results

def main(model_path, data_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model in 16-bit precision
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    
    data = load_data(data_path)
    results = generate_responses(model, tokenizer, data, device)
    save_results(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate base model outputs for patient questions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated responses.")
    args = parser.parse_args()

    main(args.model_path, args.data_path, args.output_path)
