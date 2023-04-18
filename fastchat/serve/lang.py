from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain import PromptTemplate, LLMChain
import torch

tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")

base_model = LlamaForCausalLM.from_pretrained(
    "chavinlo/alpaca-native",
    load_in_8bit=True,
    device_map='auto',
)

pipe = pipeline(
    "text-generation",
    model=base_model, 
    tokenizer=tokenizer, 
    max_length=256,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = "What is the capital of England?"

print(llm_chain.run(question))