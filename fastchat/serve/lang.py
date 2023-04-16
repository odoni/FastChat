from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

model_id = 'lmsys/vicuna-7b-delta-v0'# go for a smaller model if you dont have the VRAM
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')

pipe = pipeline(
    "text-generation",
    model=model, 
    tokenizer=tokenizer, 
    max_length=512
)

local_llm = HuggingFacePipeline(pipeline=pipe)

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = "What is the capital of England?"

print(llm_chain.run(question))