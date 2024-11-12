import csv
import time
import json
import requests

# Measure model loading time
start_loading_time = time.time()
loading_time = time.time() - start_loading_time  # Calculate loading time


def evaluate_conversations(messages, output_file):
    # Open CSV file to record inference metrics
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Message ID', 'Input Question', 'Gemma Response', 'Inference Time (seconds)', 'Throughput (tokens/second)', 'Loading Time (seconds)', 'Timestamp'])

        # Iterate through each message
        for message_index, message in enumerate(messages):
            if isinstance(message, str):  # Ensure message is a string
                start_time = time.time()
                try:
                    # Construct the request payload
                    payload = {
                        "messages": [
                            {"role": "system", "content": "You are an excellent chat assistant."},
                            {"role": "user", "content": message}
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
                        "http://10.41.119.56:30544/v1/chat/completions",
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

                end_time = time.time()  # Calculate inference time in seconds
                inference_time = end_time - start_time

                # Calculate throughput (number of tokens processed per second)
                tokens_generated = len(message.split()) + len(str(response_text).split())  # Count tokens in the response
                throughput = tokens_generated / inference_time if inference_time > 0 else 0  # Calculate throughput

                # Get the current timestamp
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # Format timestamp

                # Record the message ID, Input Question, Gemma response, inference time, throughput, loading time, and timestamp in the CSV
                csv_writer.writerow([f"{message_index}", message, response_text, inference_time, throughput, loading_time, timestamp])
            else:
                print(f"Skipping non-string message: {message}")  # Log non-string messages


# Load messages from input_sustained.json
with open('./prompts/input_sustained.json', 'r') as f:
    sustained_conversations = json.load(f)

# Flatten the sustained conversations into a single list of messages
flat_sustained_messages = [message for conversation in sustained_conversations for message in conversation]

# Call the function with the flat sustained message list
evaluate_conversations(flat_sustained_messages, './results/inference_metrics_sustained.csv')

# Load messages from conversations.json
with open('./prompts/conversations.json', 'r') as f:
    conversations = json.load(f)

# Flatten the conversations into a single list of messages
flat_messages = [message for conversation in conversations for message in conversation]

# Call the function with the flat message list for conversations.json
evaluate_conversations(flat_messages, './results/inference_metrics_conversations.csv')

# Prevent program from exiting
while True:
    pass
