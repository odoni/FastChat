from langchain.agents import create_csv_agent
from langchain.llms import OpenAI


def load_openai_agent():
    return create_csv_agent(OpenAI(temperature=0), '/home/odoni/vicuna/FastChat/fastchat/serve/factor.csv', verbose=False)