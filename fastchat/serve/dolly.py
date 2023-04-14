import torch
import instruct_pipeline
from transformers import pipeline
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map="auto")

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

generate_text("Explain to me the difference between nuclear fission and fusion.")
