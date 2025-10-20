# -*- coding: utf-8 -*-
"""
Chat interativo no terminal com modelo consolidado em .safetensors
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Caminho do modelo consolidado
model_dir = "./final_model_single"

print("🔹 Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("🔹 Carregando modelo consolidado...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",         # envia para GPU se disponível
    torch_dtype=torch.float16, # usa FP16 (menos memória)
    trust_remote_code=True
)
model.eval()

# Função de inferência
def generate_text(prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Loop de chat
if __name__ == "__main__":
    print("💬 Chat iniciado! Digite sua pergunta (ou 'sair' para encerrar).")
    while True:
        user_input = input("📝 Você: ")
        if user_input.strip().lower() in ["sair", "exit", "quit"]:
            print("👋 Encerrando o chat.")
            break
        resposta = generate_text(user_input)
        print("🤖 Modelo:", resposta)
