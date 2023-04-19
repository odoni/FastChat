from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from pydantic import BaseModel
from queue import Queue
from threading import Thread
import time
from .lang_load_local import load_local_agent

app = FastAPI()
local_queue = Queue()
openai_queue = Queue()
local_agent = load_local_agent()

class Message(BaseModel):
    message: str
    model: str

class Answer(BaseModel):
    message: str
    answer: str
    model: str

def process_message(message: str, model: str) -> str:
    # Add your message processing logic here
    # time.sleep(2)  # Simulate processing time
    return f"Processed: {message} using {model}"

def local_worker():
    while True:
        message_obj = local_queue.get()
        if message_obj is None:
            break
        message_obj["answer"] = local_agent.run(message_obj["message"])
        message_obj["processed"] = True

def openai_worker():
    while True:
        message_obj = openai_queue.get()
        if message_obj is None:
            break
        answer = local_agent.run(message_obj["message"])
        message_obj["answer"] = answer
        message_obj["processed"] = True

local_worker_thread = Thread(target=local_worker, daemon=True)
openai_worker_thread = Thread(target=openai_worker, daemon=True)
local_worker_thread.start()
openai_worker_thread.start()

@app.get("/")
async def get_chat_html():
    return FileResponse("/home/odoni/vicuna/FastChat/fastchat/serve/templates/chat.html")

@app.get("/factor.csv")
async def get_chat_html():
    return FileResponse("/home/odoni/vicuna/FastChat/fastchat/serve/factor.csv")

@app.post("/message", response_model=Answer)
async def post_message(message_obj: Message):
    message_dict = message_obj.dict()
    message_dict["processed"] = False

    if message_dict["model"] == "local":
        local_queue.put(message_dict)
    elif message_dict["model"] == "openai":
        openai_queue.put(message_dict)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    while not message_dict["processed"]:
        time.sleep(0.1)

    return Answer(**message_dict)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
