from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Define the model name for GPT-Neo 125M
model_name = "EleutherAI/gpt-neo-125m"
model_path = 'gpt_neo'


# Load the tokenizer and model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path) #return_dict=True, trust_remote_code=True, safe_serialization=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)

#model.save_pretrained(model_path)
#tokenizer.save_pretrained(model_path)

# Sample input for conversation
input_text = "Hello, when and where the first olympic games happened?"

# Tokenize the input
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate the model's response
output_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

# Decode and print the response
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Response from the model:")
print(response)

