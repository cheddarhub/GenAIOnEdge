from ctransformers import AutoModelForCausalLM
import time

#TheBloke/Llama-2-7b-Chat-GGUF
# pip install ctransformers
# download:  huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("./model", model_file="llama-2-7b-chat.Q4_K_M.gguf", model_type="llama", gpu_layers=0)

start_time = time.time()
print(llm("AI is going to"))

end_time = time.time()# Calculate inference time in seconds
inference_time = end_time - start_time
print(f"Inference Latency: {inference_time} seconds")
