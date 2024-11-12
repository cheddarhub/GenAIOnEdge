from typing import Union

from fastapi import FastAPI, Body

app = FastAPI()

from ctransformers import AutoModelForCausalLM
import time

llm = AutoModelForCausalLM.from_pretrained("./model", model_file="llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", gpu_layers=0)

@app.get("/")
def read_root():
    return {"Hello": "World llama"}


@app.post("/chat")
def chat_post(message: str = Body(..., embed=True)):
    
    start_time = time.time()
    response = llm(message)
    end_time = time.time()# Calculate inference time in seconds
    inference_time = end_time - start_time
    # Append inference time to the response
    response_with_time = f"{response} \n (Inference time: {inference_time:.4f} seconds)"
    # Return the response
    return {
        "response": response_with_time,
        "inference_time": f"{inference_time:.4f}"
    }
    