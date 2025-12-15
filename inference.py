# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Caminho do modelo base (sem fine-tuning)
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Configuração de quantização 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Carrega tokenizer do modelo base
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Carrega modelo base com quantização 4-bit
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Função para gerar resposta
def gerar_resposta(prompt, max_tokens=2000):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.1
        )
    resposta = tokenizer.decode(output[0], skip_special_tokens=True)
    return resposta

# Exemplo de uso
if __name__ == "__main__":
    while True:
        prompt = input("\nDigite sua pergunta (ou 'sair'): ")
        if prompt.lower() == "sair":
            break
        resultado = gerar_resposta(prompt)
        print("\n📎 Resposta gerada:")
        print(resultado)
