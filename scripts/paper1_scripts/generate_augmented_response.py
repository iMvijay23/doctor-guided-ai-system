import os
#os.environ['HF_HOME'] = '/scratch4/mdredze1/vtiyyal1/huggingface_cache/'
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
from huggingface_hub import HfFolder

# Set HuggingFace cache directory

HfFolder.save_token(os.getenv('HUGGINGFACE_TOKEN'))

def load_data(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)
    return data

def save_result(result, filepath):
    with open(filepath, 'a') as file:
        json.dump(result, file)
        file.write('\n')

def generate_responses(model, tokenizer, data, device, output_path, batch_size=1):
    for item in tqdm(data):
        query = item['question']
        doctor_response = item['answer']
        prompt = f"###Instruction: Generate a comprehensive patient-friendly response that includes all the key points and recommendations from the doctor's response. Ensure that no important details from the doctor's response are omitted.\n###Patient Query: {query}\n###Doctor's Response: {doctor_response}\n###AI-Augmented Response:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract text after "###AI-Augmented Response:"
        if "###AI-Augmented Response:" in response:
            response = response.split("###AI-Augmented Response:")[1].strip()
        
        result = {
            'primaryid': item.get('primaryid', None),
            'question': query,
            'doctor_response': doctor_response,
            'ai_augmented_response': response
        }
        
        save_result(result, output_path)

def main(model_path, data_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model in 16-bit precision
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    
    data = load_data(data_path)
    generate_responses(model, tokenizer, data, device, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI-augmented model outputs for patient questions.")
    parser.add_argument("--model_path", type=str, default='/home/vtiyyal1/scratch4-mdredze1/huggingface_cache/transformers', help="Path to the pretrained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated responses.")
    args = parser.parse_args()

    main(args.model_path, args.data_path, args.output_path)