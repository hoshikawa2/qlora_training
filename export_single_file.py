# -*- coding: utf-8 -*-
"""
Exporta modelo LoRA fundido em um único arquivo .safetensors + tokenizer
"""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Diretórios
src_dir = "./merged_model"          # onde já está o modelo fundido (shards)
dst_dir = "./final_model_single"    # destino consolidado
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

os.makedirs(dst_dir, exist_ok=True)

print("🔹 Carregando modelo fundido...")
model = AutoModelForCausalLM.from_pretrained(
    src_dir,
    torch_dtype="auto",   # mais estável que float16 fixo
    device_map="cpu"      # força carregar em CPU (não estoura GPU)
)

print("🔹 Salvando em safetensors único...")
model.save_pretrained(
    dst_dir,
    safe_serialization=True,   # gera model.safetensors
    max_shard_size="30GB"      # ajusta para caber tudo em 1 arquivo
)

print("🔹 Salvando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(dst_dir)

print(f"✅ Arquivo único salvo em {dst_dir}/model.safetensors")
