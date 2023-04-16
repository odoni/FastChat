from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
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
    max_length=2054,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.2
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# template = """A chat between a curious human and an artificial intelligence assistant. The artificial intelligence assistant name is GlobosoGPT. The assistant gives helpful, detailed, and polite answers to the human's questions.
# Human: What are the key differences between renewable and non-renewable energy sources?
# Renewable energy sources are those that can be replenished naturally in a relatively 
# short amount of time, such as solar, wind, hydro, geothermal, and biomass. 
# Non-renewable energy sources, on the other hand, are finite and will eventually be 
# depleted, such as coal, oil, and natural gas. Here are some key differences between 
# renewable and non-renewable energy sources:\n
# 1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable 
# energy sources are finite and will eventually run out.\n
# 2. Environmental impact: Renewable energy sources have a much lower environmental impact 
# than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, 
# and other negative effects.\n
# 3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically 
# have lower operational costs than non-renewable sources.\n
# 4. Reliability: Renewable energy sources are often more reliable and can be used in more remote 
# locations than non-renewable sources.\n
# 5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different 
# situations and needs, while non-renewable sources are more rigid and inflexible.\n
# 6. Sustainability: Renewable energy sources are more sustainable over the long term, while 
# non-renewable sources are not, and their depletion can lead to economic and social instability.\n
# Human: {instruction}
# Assistant: 
# """

template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: 
{instruction}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["instruction"])

print("\n\n\n" + prompt + "\n\n\n")

llm_chain = LLMChain(prompt=prompt, 
                     llm=local_llm
                     )

question = "What is the capital of England?"

print(llm_chain.run(question))