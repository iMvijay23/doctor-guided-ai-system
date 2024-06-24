import os
import json
import torch
import openai
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PeftModelForCausalLM


def load_config():
    with open('config.json') as f:
        return json.load(f)

config = load_config()

os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
os.environ["HUGGINGFACE_API_KEY"] = config["HUGGINGFACE_API_KEY"]

# Global model paths
MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"
EMPATHY_MODEL_PATH = "/scratch4/mdredze1/vtiyyal1/models/askdocsproject/checkpoints_apr3/llama2chat/checkpoint-5999"
REPHRASING_MODEL_PATH = "meta-llama/Llama-2-7b-chat-hf"

# Data path
DATA_PATH = "/home/vtiyyal1/data-mdredze1/vtiyyal1/askdocschat/high_quality_long_answers_data_apr10.json"
RESULTS_PATH = "/home/vtiyyal1/data-mdredze1/vtiyyal1/askdocschat/doctor-guided-system/dgs_trial1_results.json"

class FactEvaluator:
    def __init__(self, openai_key):
        self.openai_key = openai_key
        openai.api_key = self.openai_key

    def generate_atomic_facts(self, text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Extract atomic facts from this text: {text}"}
            ],
            max_tokens=150
        )
        raw_facts = response.choices[0].message['content'].strip()
        facts = re.split(r'\n|;', raw_facts)
        facts = [fact.strip() for fact in facts if fact.strip()]
        return facts

    def compare_facts_with_reference(self, fact, reference):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Does the statement: '{fact}' logically follow from the reference: '{reference}'? Answer True or False."}
            ],
            max_tokens=10
        )
        return response.choices[0].message['content'].strip().lower() == "true"

    def evaluate_hallucination_level(self, generation, reference):
        gen_facts = self.generate_atomic_facts(generation)
        ref_facts = self.generate_atomic_facts(reference)
        
        comparison_results = []
        for gen_fact in gen_facts:
            is_supported = any(self.compare_facts_with_reference(gen_fact, ref_fact) for ref_fact in ref_facts)
            comparison_results.append({"fact": gen_fact, "supported": is_supported})
        
        overall_score = sum(result["supported"] for result in comparison_results) / len(comparison_results) if comparison_results else 0
        return overall_score, comparison_results

class DoctorGuidedSystem:
    def __init__(self, llm, empathy_model_path=None, medical_db_api=None, device="cuda", use_quantize=False):
        self.llm = llm
        self.empathy_model_path = empathy_model_path
        self.medical_db_api = medical_db_api
        self.device = device
        self.use_quantize = use_quantize

        if self.llm == "openai":
            openai.api_key = os.environ["OPENAI_API_KEY"]
        elif self.llm == "local":
            self.model = self.load_model(MODEL_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.empathy_model = self.load_empathy_model(EMPATHY_MODEL_PATH) if empathy_model_path else None

    def load_model(self, model_path):
        if self.use_quantize:
            double_quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=double_quant_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        model.to(self.device)
        return model

    def load_empathy_model(self, empathy_model_path):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
        empathy_model = PeftModelForCausalLM.from_pretrained(base_model, empathy_model_path)
        return empathy_model.merge_and_unload().half().to(self.device)

    def generate(self, prompt, **kwargs):
        if self.llm == "openai":
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return completion.choices[0].message['content'].strip()
        elif self.llm == "local":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def process_query_response(self, patient_query, doctor_response):
        return {
            'query': patient_query,
            'response': doctor_response
        }

    def breakdown_medical_advice(self, medical_advice):
        prompt = f"Break down the following medical advice into key facts or claims. List each fact or claim on a new line:\nMedical Advice: {medical_advice}"
        key_facts = self.generate(prompt, max_tokens=150)
        return key_facts

    def generate_expanded_response(self, question, key_facts, original_response):
        prompt = f"Using the following key facts derived from a doctor's response, generate a comprehensive response to the patient's question. Ensure the expanded response is detailed and easy to understand, without changing the original medical advice.\nKey Facts: {key_facts}\nOriginal Response: {original_response}\nPatient Query: {question}"
        expanded_response = self.generate(prompt, max_tokens=300)
        return expanded_response

    def verify_claims(self, response, original_response):
        prompt = f"Compare the following claims with the original doctor's advice. For each claim, indicate whether it is supported by the original advice or not. Provide a detailed breakdown.\nOriginal Doctor's Advice: {original_response}\nClaims: {response}"
        verified_response = self.generate(prompt, max_tokens=500)
        
        # Parse the verified response to create a structured output
        claims = re.split(r'\n(?=\d+\.)', verified_response)
        structured_claims = []
        for claim in claims:
            if claim.strip():
                parts = claim.split(':')
                if len(parts) >= 2:
                    claim_text = ':'.join(parts[1:]).strip()
                    is_supported = "supported" in claim_text.lower()
                    structured_claims.append({
                        "claim": claim_text,
                        "supported": is_supported
                    })
        
        return structured_claims

    def inject_empathy(self, question, verified_claims):
        # Combine verified claims into a single response
        verified_response = " ".join([claim["claim"] for claim in verified_claims if claim["supported"]])
    
        prompt = f"""<s>[INST] <<SYS>> ###System: You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Additionally, the goal is to augment the empathy in medical responses without altering any factual medical content. For context, here is the question related to the medical response: <</SYS>>[INST] ###Question: {question} And here is the original response that needs to be more empathetic: ###Answer: {verified_response} [/INST] ###Empathetic Response:"""
    
        empathetic_response = self.generate(prompt, max_tokens=256)
        return empathetic_response.strip()

    def evaluate_response(self, response, original_response):
        factuality_score, fact_comparison = self.evaluate_factuality(response, original_response)
        quality_score = self.evaluate_quality(response)
        empathy_score = self.evaluate_empathy(response)
        return {
            'factuality': factuality_score,
            'quality': quality_score,
            'empathy': empathy_score,
            'fact_comparison': fact_comparison
        }

    def evaluate_factuality(self, response, original_response):
        evaluator = FactEvaluator(openai_key=os.environ["OPENAI_API_KEY"])
        return evaluator.evaluate_hallucination_level(response, original_response)

    def evaluate_quality(self, response):
        # Placeholder implementation
        return len(response.split()) / 100

    def evaluate_empathy(self, response):
        # Placeholder implementation
        return len([word for word in response.split() if word in ['feel', 'understand', 'sorry']]) / len(response.split())

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def save_result(result, filepath):
    with open(filepath, 'a') as file:
        json.dump(result, file)
        file.write('\n')

def run_pipeline(system, data, batch_size, llm="openai"):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        for item in batch:
            query = item['question']
            response = item['answer']
            processed_data = system.process_query_response(query, response)
            key_facts = system.breakdown_medical_advice(processed_data['response'])
            expanded_response = system.generate_expanded_response(processed_data['query'], key_facts, processed_data['response'])
            verified_claims = system.verify_claims(expanded_response, processed_data['response'])
            verified_claims_response = " ".join([claim["claim"] for claim in verified_claims if claim["supported"]])
            
            empathetic_response = system.inject_empathy(query, verified_claims)
            #scores = system.evaluate_response(empathetic_response, processed_data['response'])
            
            result = {
                'primaryid': item['primaryid']
                'query': query,
                'original_response': response,
                'key_facts': key_facts,
                'expanded_response': expanded_response,
                'verified_claims': verified_claims,
                'verified_claims_response': verified_claims_response,
                'empathetic_response': empathetic_response,
                #'scores': scores
            }
            
            save_result(result, RESULTS_PATH)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a model.")
    parser.add_argument("--use_quantize", type=int, default=0, help="Use quantization if set to 1.")
    args = parser.parse_args()

    llm = "local"  # or "openai"
    empathy_model = EMPATHY_MODEL_PATH
    system = DoctorGuidedSystem(llm, empathy_model_path=empathy_model, medical_db_api=None, device="cuda", use_quantize=bool(args.use_quantize))

    data = load_data(DATA_PATH)
    run_pipeline(system, data, batch_size=2, llm=llm)