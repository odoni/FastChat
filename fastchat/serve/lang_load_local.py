from langchain.agents import create_csv_agent
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch

def load_local_agent():
    # model_id = "chavinlo/alpaca-native"
    model_id = "chavinlo/alpaca-native"
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    base_model = LlamaForCausalLM.from_pretrained(
        model_id,
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

    template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction: 
    {instruction}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["instruction"])

    return create_csv_agent(llm, '/home/odoni/vicuna/FastChat/fastchat/serve/factor.csv', prompt=prompt)