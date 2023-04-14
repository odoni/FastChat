import torch
from transformers import pipeline
from .instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-7b", padding_side="left", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-7b", device_map="auto", torch_dtype=torch.bfloat16)

print("Loading text generation")
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

print(generate_text("Explain to me the difference between nuclear fission and fusion."))
