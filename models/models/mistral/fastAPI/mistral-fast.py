from typing import Union

from fastapi import FastAPI, Body

app = FastAPI()

import time
from llama_cpp import Llama
import csv  # Import csv for file writing
import os  # Import os for directory handling

# Measure model loading time
start_loading_time = time.time()
llm = Llama(model_path="./model/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf")
loading_time = time.time() - start_loading_time
print(f"Model loading time: {loading_time:.4f} seconds")  # Log loading time

csv_file_path = "/data/mistral_model_performance_data.csv"  # Update to the mounted path

# Ensure the directory exists
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)  # Create directory if it doesn't exist
print(f"CSV file will be created at: {csv_file_path}")  # Print message about the file creation

@app.get("/")
def read_root():
    return {"Hello": "World mistral"}

@app.post("/chat")
async def chat(message: str = Body(..., embed=True)):
    start_time = time.time()
    
    # Create chat completion
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )
    
    # Calculate inference time
    inference_time = time.time() - start_time
    # Extract the assistant's full response
    response_full = response['choices'][0]['message']['content']
    
    # Calculate number of tokens including the full response
    num_tokens = len(message.split()) + len(response_full.split())  # Updated to use response_full
    throughput = num_tokens / inference_time if inference_time > 0 else 0
    
    response_with_time = f"{response_full} \n (Inference time: {inference_time:.4f} seconds, Throughput: {throughput:.2f} tokens/second, Loading time: {loading_time:.4f} seconds)"
    
    # Export data to a CSV file
    current_time = time.time()  # Get the current time for the CSV entry
    with open(csv_file_path, mode='a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([current_time, loading_time, inference_time, throughput])  # Write the data row
    
    return {
        "response": response_with_time,
        "inference_time": inference_time,
        "throughput": throughput,
        "loading_time": loading_time  # Add loading time to the response
    }