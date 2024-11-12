from typing import Union

from fastapi import FastAPI, Body

app = FastAPI()
from llama_cpp import Llama
import time

# Load the model
model_path = "./model/internlm2_5-7b-chat-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=2048, n_batch=512)

@app.get("/")
def read_root():
    return {"Hello": "World internLM"}

@app.post("/chat")
def chat_post(message: str = Body(..., embed=True)):
    start_time = time.time()
    response = llm(message, max_tokens=200)
    end_time = time.time()
    inference_time = end_time - start_time
    
    response_with_time = f"{response['choices'][0]['text']} \n (Inference time: {inference_time:.4f} seconds)"
    return {
        "response": response_with_time,
        "inference_time": f"{inference_time:.4f}"
    }
