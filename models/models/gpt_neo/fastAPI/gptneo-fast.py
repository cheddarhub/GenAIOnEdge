from typing import Union

from fastapi import FastAPI, Body

app = FastAPI()

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
# Define the model name for GPT-Neo 125M
model_name = "EleutherAI/gpt-neo-125m"
model_path = './model'


# Load the tokenizer and model
model = AutoModelForCausalLM.from_pretrained(model_path) #return_dict=True, trust_remote_code=True, safe_serialization=False)
tokenizer = AutoTokenizer.from_pretrained(model_path)


@app.get("/")
def read_root():
    return {"Hello": "World gpt-neo"}

@app.get("/chat/{message}")
def chat(message: str):
    # Tokenize the message
    input_ids = tokenizer.encode(message, return_tensors="pt")

    # Generate the model's response
    output_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

    # Decode and print the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return f"response is {response}"

@app.post("/chat")
def chat_post(message: str = Body(..., embed=True)):
    # Tokenize the message
    input_ids = tokenizer.encode(message, return_tensors="pt")

    # Generate the model's response
    start_time = time.time()
    output_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    end_time = time.time()
    inference_time = end_time - start_time
    # Decode the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    response_with_time = f"{response} \n (Inference time: {inference_time:.4f} seconds)"
    # Return the response
    return {
        "response": response_with_time,
        "inference_time": f"{inference_time:.4f}"
    }# Return the response
    #return {"response": response, "inference_time": inference_time}
    
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}