import os
import json
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import torch
import openai
from peft import PeftModel, PeftConfig, PeftModelForCausalLM

def load_config():
    with open('/home/vtiyyal1/askdocs/Doctor-Guided-AI/config.json') as f:
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
RESULTS_PATH = "/home/vtiyyal1/data-mdredze1/vtiyyal1/askdocschat/doctor-guided-system/dgs_results_trial8_ica.json"

class FactEvaluatorOpenAI:
    def __init__(self, threshold_score=7.0, min_avg_score=5.0, min_supported_percentage=0.5):
        #openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.threshold_score = threshold_score
        self.min_avg_score = min_avg_score
        self.min_supported_percentage = min_supported_percentage

    def generate_atomic_facts(self, text):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You will break down the text into atomic facts."},
                {"role": "user", "content": f"Extract atomic facts from the following text:\n{text}"}
            ],
            max_tokens=150
        )
        raw_facts = response.choices[0].message.content.strip()
        facts = re.split(r'\n|;', raw_facts)
        facts = [fact.strip() for fact in facts if fact.strip()]
        return facts

    def compare_facts_with_reference(self, fact, reference):
        prompt = (f"Evaluate the following statement for its support by the given reference on a scale from 0 to 10, where 0 means 'not supported at all' and 10 means 'fully supported'. "
          f"Only provide the numeric score.\n\n"
          f"Statement: '{fact}'\n"
          f"Reference: '{reference}'\n"
          f"Score:")
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You will assess factual consistency."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=15
        )
        score = response.choices[0].message.content.strip()
        try:
            return float(score)
        except ValueError:
            return 0.0

    def evaluate_hallucination_level(self, generation, reference):
        gen_facts = self.generate_atomic_facts(generation)
        fact_scores = {}
        supported_count = 0
        for fact in gen_facts:
            score = self.compare_facts_with_reference(fact, reference)
            fact_scores[fact] = score
            if score >= self.threshold_score:
                supported_count += 1

        overall_score = np.mean(list(fact_scores.values())) if fact_scores else 0
        supported_percentage = supported_count / len(gen_facts) if gen_facts else 0
        passes_test = bool(overall_score >= self.min_avg_score and supported_percentage >= self.min_supported_percentage)

        return overall_score, fact_scores, passes_test

    def evaluate_hallucination_level_facts(self, gen_facts, reference):
        fact_scores = {}
        supported_count = 0
        for fact in gen_facts:
            score = self.compare_facts_with_reference(fact, reference)
            fact_scores[fact] = score
            if score >= self.threshold_score:
                supported_count += 1

        overall_score = np.mean(list(fact_scores.values())) if fact_scores else 0
        supported_percentage = supported_count / len(gen_facts) if gen_facts else 0
        passes_test = bool(overall_score >= self.min_avg_score and supported_percentage >= self.min_supported_percentage)

        return overall_score, fact_scores, passes_test

class FactEvaluatorLocal:
    def __init__(self, model, tokenizer, device="cuda", threshold_score=7.0, min_avg_score=5.0, min_supported_percentage=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold_score = threshold_score
        self.min_avg_score = min_avg_score
        self.min_supported_percentage = min_supported_percentage

    def generate_atomic_facts(self, text):
        prompt = f"Extract atomic facts from the following text:\n{text}\n###Atomic Facts:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        raw_facts = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # Extract facts after the tag
        if "###Atomic Facts:" in raw_facts:
            raw_facts = raw_facts.split("###Atomic Facts:")[1].strip()
        facts = re.split(r'\n|;', raw_facts)
        facts = [fact.strip() for fact in facts if fact.strip()]
        return facts

    def compare_facts_with_reference(self, fact, reference):
        prompt = (f"Evaluate the following statement for its support by the given reference on a scale from 0 to 10, where 0 means 'not supported at all' and 10 means 'fully supported'. "
          f"Only provide the numeric score.\n\n"
          f"Statement: '{fact}'\n"
          f"Reference: '{reference}'\n"
          f"Score:")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=15)
        score = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        try:
            return float(score)
        except ValueError:
            return 0.0

    def evaluate_hallucination_level(self, generation, reference):
        gen_facts = self.generate_atomic_facts(generation)
        fact_scores = {}
        supported_count = 0
        for fact in gen_facts:
            score = self.compare_facts_with_reference(fact, reference)
            fact_scores[fact] = score
            if score >= self.threshold_score:
                supported_count += 1

        overall_score = np.mean(list(fact_scores.values())) if fact_scores else 0
        supported_percentage = supported_count / len(gen_facts) if gen_facts else 0
        passes_test = bool(overall_score >= self.min_avg_score and supported_percentage >= self.min_supported_percentage)

        return overall_score, fact_scores, passes_test

    def evaluate_hallucination_level_facts(self, gen_facts, reference):
        fact_scores = {}
        supported_count = 0
        for fact in gen_facts:
            score = self.compare_facts_with_reference(fact, reference)
            fact_scores[fact] = score
            if score >= self.threshold_score:
                supported_count += 1

        overall_score = np.mean(list(fact_scores.values())) if fact_scores else 0
        supported_percentage = supported_count / len(gen_facts) if gen_facts else 0
        passes_test = bool(overall_score >= self.min_avg_score and supported_percentage >= self.min_supported_percentage)

        return overall_score, fact_scores, passes_test

class DoctorGuidedSystem:
    def __init__(self, llm, empathy_model_path=None, medical_db_api=None, device="cuda", use_quantize=False):
        self.llm = llm
        self.empathy_model_path = empathy_model_path
        self.medical_db_api = medical_db_api
        self.device = device
        self.use_quantize = use_quantize

        if self.llm == "openai":
            self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        peftconfig = PeftConfig.from_pretrained(empathy_model_path)
        empathy_model = AutoModelForCausalLM.from_pretrained(peftconfig.base_model_name_or_path, return_dict=True, quantization_config=bnb_config, device_map="auto")
        return empathy_model

    def generate(self, prompt, **kwargs):
        if self.llm == "openai":
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return completion.choices[0].message.content.strip()
        elif self.llm == "local":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, **kwargs)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def generate_with_empathy_model(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.empathy_model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


    def process_query_response(self, patient_query, doctor_response):
        return {
            'query': patient_query,
            'response': doctor_response
        }

    def breakdown_medical_advice(self, medical_advice):
        prompt = f"Break down the following medical advice into key facts or claims. Exclude any non-medical advice. List each medical fact or claim on a new line, starting with a dash (-). Do not include any other text:\nMedical Advice: {medical_advice}\n###Key Facts:"
        key_facts = self.generate(prompt, max_new_tokens=256)
        return self.post_process_key_facts(key_facts, medical_advice)

    def post_process_key_facts(self, key_facts, original_text):
        if "Key Facts:" in key_facts:
            key_facts = key_facts.split("Key Facts:")[1].strip()
        
        overlap = self.find_overlap(original_text, key_facts)
        if overlap:
            key_facts = key_facts[len(overlap):].strip()
        
        facts_list = [fact.strip().lstrip('- ') for fact in key_facts.split('\n') if fact.strip()]
        
        return facts_list

    def find_overlap(self, str1, str2):
        # Find the longest overlap between the end of str1 and the start of str2
        for i in range(min(len(str1), len(str2)), 0, -1):
            if str1[-i:] == str2[:i]:
                return str2[:i]
        return ""

    def generate_expanded_response(self, question, key_facts, original_response):
        prompt = f"Generate a comprehensive and detailed response to the patient's question using only the key facts listed below. Ensure that the expanded response is understandable, adheres strictly to these key facts.\nKey Facts:\n{key_facts}\nPatient Query: {question}\n###Expanded Response:"
        expanded_response = self.generate(prompt, max_new_tokens=256)
        return expanded_response.split("###Expanded Response:")[1].strip()

    def inject_empathy_claims(self, question, verified_claims):
        # Combine verified claims into a single response
        verified_response = " ".join([claim for claim in verified_claims if verified_claims[claim] >= 7.0])    
        prompt = f"""###System: You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Additionally, the goal is to augment the empathy in medical responses without altering any factual medical content. For context, here is the question related to the medical response:\n###Question: {question}\nAnd here is the original response that needs to be more empathetic:\n###Answer: {verified_response}\n###Empathetic Response:"""
    
        empathetic_response = self.generate_with_empathy_model(prompt, max_new_tokens=512).strip()
        # Extract text after "###Empathetic Response:"
        if "###Empathetic Response:" in empathetic_response:
            empathetic_response = empathetic_response.split("###Empathetic Response:")[1].strip()

        return empathetic_response

    def inject_empathy(self, question, verified_response):
          
        prompt = f"""###System: You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Additionally, the goal is to augment the empathy in medical responses without altering any factual medical content. For context, here is the question related to the medical response:\n###Question: {question}\nAnd here is the original response that needs to be more empathetic:\n###Answer: {verified_response}\n###Empathetic Response:"""
    
        empathetic_response = self.generate_with_empathy_model(prompt, max_new_tokens=512).strip()
        # Extract text after "###Empathetic Response:"
        if "###Empathetic Response:" in empathetic_response:
            empathetic_response = empathetic_response.split("###Empathetic Response:")[1].strip()

        return empathetic_response

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

    def evaluate_factuality(self, response, original_response, use_openai=False):
        if use_openai:
            evaluator = FactEvaluatorOpenAI(threshold_score=7.0, min_avg_score=5.0, min_supported_percentage=0.5)
        else:
            evaluator = FactEvaluatorLocal(self.model, self.tokenizer, self.device, threshold_score=7.0, min_avg_score=5.0, min_supported_percentage=0.5)
        return evaluator.evaluate_hallucination_level(response, original_response)

    def evaluate_factuality_facts(self, response, original_response, use_openai=False):
        if use_openai:
            evaluator = FactEvaluatorOpenAI(threshold_score=7.0, min_avg_score=5.0, min_supported_percentage=0.5)
        else:
            evaluator = FactEvaluatorLocal(self.model, self.tokenizer, self.device, threshold_score=7.0, min_avg_score=5.0, min_supported_percentage=0.5)
        return evaluator.evaluate_hallucination_level_facts(response, original_response)

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

def run_pipeline(system, data, batch_size, llm="openai", use_openai_for_facts=False):
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        for item in batch:
            query = item['question']
            response = item['answer']
            processed_data = system.process_query_response(query, response)
            # Original response key facts
            original_key_facts = system.breakdown_medical_advice(processed_data['response'])
            
            # Generate expanded response
            expanded_response = system.generate_expanded_response(processed_data['query'], "\n".join(original_key_facts), processed_data['response'])
            
            # Extract key facts from expanded response
            expanded_key_facts = system.breakdown_medical_advice(expanded_response)
            
            # Evaluate factuality of expanded key facts against original response
            overall_score, fact_scores, passes_test = system.evaluate_factuality_facts(expanded_key_facts, processed_data['response'], use_openai=use_openai_for_facts)
            
            # Filter high-scoring facts
            supported_facts = {fact: score for fact, score in fact_scores.items() if score >= 7.0}
            
            # Generate intermediary response using supported facts
            intermediary_response = system.generate_expanded_response(processed_data['query'], "\n".join(supported_facts.keys()), processed_data['response'])
            
            # Inject empathy into intermediary response
            empathetic_response = system.inject_empathy(query, intermediary_response)
            
            # Extract facts from empathetic response
            empathetic_key_facts = system.breakdown_medical_advice(empathetic_response)
            
            result = {
                'primaryid': item['primaryid'],
                'query': query,
                'original_response': response,
                'original_key_facts': original_key_facts,
                'expanded_response': expanded_response,
                'expanded_key_facts': expanded_key_facts,
                'fact_scores': fact_scores,
                'supported_facts': supported_facts,
                'intermediary_response': intermediary_response,
                'empathetic_response': empathetic_response,
                'empathetic_key_facts': empathetic_key_facts,
                #'passes_test': passes_test,
                #'overall_score': overall_score
            }
            
            save_result(result, RESULTS_PATH)
        #break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on a model.")
    parser.add_argument("--use_quantize", type=int, default=0, help="Use quantization if set to 1.")
    parser.add_argument("--use_openai_for_facts", type=bool, default=False, help="Use OpenAI for fact-checking if set to True.")
    args = parser.parse_args()

    llm = "local"  # or "openai"
    empathy_model = EMPATHY_MODEL_PATH
    system = DoctorGuidedSystem(llm, empathy_model_path=empathy_model, medical_db_api=None, device="cuda", use_quantize=bool(args.use_quantize))

    data = load_data(DATA_PATH)
    run_pipeline(system, data, batch_size=2, llm=llm, use_openai_for_facts=args.use_openai_for_facts)
