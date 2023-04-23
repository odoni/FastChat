from langchain.agents import create_csv_agent
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch

def load_local_agent():
    model = "chavinlo/alpaca-native"
    # model = "lmsys/vicuna-7b-delta-v0"
    tokenizer = LlamaTokenizer.from_pretrained(model)

    base_model = LlamaForCausalLM.from_pretrained(
        model,
        load_in_8bit=True,
        device_map='auto',
    )

    pipe = pipeline(
        "text-generation",
        model=base_model, 
        tokenizer=tokenizer, 
        max_length=2000,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.2
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    return create_csv_agent(llm, '/home/odoni/vicuna/FastChat/fastchat/serve/factor.csv')