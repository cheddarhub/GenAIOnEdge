from transformers import GPT2Tokenizer, GPT2Model
from transformers import pipeline, set_seed

model_path = './gpt_mini'

tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2', cache_dir=model_path)
model = GPT2Model.from_pretrained('openai-community/gpt2', cache_dir=model_path)

# Sample input text
# Set pad token as eos token
tokenizer.pad_token = tokenizer.eos_token

# Sample input text
text = "Replace me by any text you'd like."

# Tokenize the input and return input_ids and attention_mask tensors
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Pass the tokenized input to the model
output = model(encoded_input)  # Unpack the tokenized inputs

# Print the model output
print(output)
