from langchain.agents import create_csv_agent
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch

def load_local_agent():
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
        max_length=512,
        temperature=0,
        top_p=0.5,
        repetition_penalty=1.2
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return create_csv_agent(llm, '/home/odoni/vicuna/FastChat/fastchat/serve/factor.csv', verbose=False, prefix="Final Answer:")