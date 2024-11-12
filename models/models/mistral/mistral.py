# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = './mistral'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

#model.save_pretrained(model_path)
#tokenizer.save_pretrained(model_path)

inputs = tokenizer("Hello my name is", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


