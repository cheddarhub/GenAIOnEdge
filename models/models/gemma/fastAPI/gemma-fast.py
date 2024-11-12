from typing import Union

from fastapi import FastAPI, Body

app = FastAPI()
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline
import time

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},  # Optional, but bfloat16 is better suited for GPU. Use float32 for CPU.
    device=-1  # Set device to -1 to run on CPU
)

@app.post("/chat")
def chat_post(message: str = Body(..., embed=True)):
    messages = [
        {"role": "user", "content": message},
    ]

    outputs = pipe(messages, max_new_tokens=256)
    assistant_response = outputs[0]["generated_text"][-1]["content"].strip()
    return {"response": assistant_response}
