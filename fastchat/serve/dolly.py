import torch
from transformers import pipeline
from .instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading model")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b", padding_side="left", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b", device_map="auto", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "/home/odoni/dolly/output/dolly2_local_v2", torch_dtype=torch.bfloat16)

print("Loading text generation")
generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

print(generate_text("Explain to me the difference between nuclear fission and fusion."))
