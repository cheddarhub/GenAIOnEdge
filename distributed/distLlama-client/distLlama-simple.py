import csv
import time
import json
import requests

message = "Hi llama, how are you?"
message_index = 1
# Print the request payload for debugging
print(f"Message ID: {message_index}")
print(f"Input Question: {message}")  # Print the input question
  
try:
    # Construct the request payload
    payload = {
        "messages": [
            {"role": "system", "content": "You are an excellent chat assistant."},
            {"role": "user", "content": message+"<|eot_id|>"}
        ],
        "temperature": 0,
        "stop": ["<|eot_id|>"],  # Include the stop parameter
        "max_tokens": 500
    }

    # Print the request payload for debugging
    print(f"Message ID: {message_index}")
    print(f"Input Question: {message}")  # Print the input question
    print(f"Payload: {json.dumps(payload, indent=4)}")  # Pretty print the payload being sent

    # POST request with proper JSON formatting and headers
    r = requests.post(
        "http://10.41.119.129:30422/v1/chat/completions",
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=None
    )

    # Check if the request was successful
    if r.status_code == 200:
        # Extract response content
        response_text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    else:
        response_text = f"Error: Received status code {r.status_code}"

    # Print the server response for debugging
    print(f"Gemma Response: {response_text}")

except Exception as e:
    print(f"Exception occurred: {str(e)}")
    response_text = ""

                
# Prevent program from exiting
while True:
    pass
