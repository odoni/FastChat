import torch
from transformers import pipeline
from .instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b", device_map="auto")
model = PeftModel.from_pretrained(model, "/home/odoni/dolly/output/dolly2_local_v2")

print("Loading text generation")
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

print(generate_text("Explain to me the difference between nuclear fission and fusion."))
