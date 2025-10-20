from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "./qlora-output"
output_path = "./merged_model"

# Carrega o modelo base
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True)

# Carrega LoRA fundido
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()

# Salva pesos fundidos em múltiplos shards
model.save_pretrained(output_path, max_shard_size="4GB")

# Copia tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
tokenizer.save_pretrained(output_path)
