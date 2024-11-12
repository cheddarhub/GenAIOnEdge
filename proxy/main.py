import requests
from typing import Union

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
#http://10.41.119.56:8082
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # traffic coming from (front ip: front port) is allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {
    "gpt_neo":"http://gpt-neo:8000",
    "llama":"http://llama:8000",
    "mistral":"http://mistral:8000",
    "internlm":"http://internlm:8000"
}
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/internlm")
def get_gpt_neo():
    try:
        response = requests.get(models["internlm"])
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to communicate with GPT-Neo: {str(e)}"}

@app.get("/mistral")
def get_gpt_neo():
    try:
        response = requests.get(models["mistral"])
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to communicate with GPT-Neo: {str(e)}"}


@app.get("/gpt-neo")
def get_gpt_neo():
    try:
        response = requests.get(models["gpt_neo"])
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to communicate with GPT-Neo: {str(e)}"}

@app.get("/llama")
def get_llama():
    try:
        response = requests.get(models["llama"])
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to communicate with Llama: {str(e)}"}


@app.post("/chat")
def chat_post(message: str = Body(..., embed=True), model: str = Body(..., embed=True)):
    
    if model not in models:
        return {"error": "Invalid model specified"}

    try:
        response = requests.post(models[model]+"/chat", json={"message": message})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Failed to communicate with the model: {str(e)}"}
    
    