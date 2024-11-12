
import json
import csv
import time
import re
import io
import sys
from llama_cpp import Llama

# Load the model
model_path = "./model/Llama-3.2-3B-Instruct-Q4_0_4_4.gguf"
start_loading_time = time.time()
llm = Llama(model_path=model_path, n_ctx=2048, n_batch=512)
end_loading_time = time.time()
model_loading_time = end_loading_time - start_loading_time
print("model loaded!")
# Regular expressions for extracting timing metrics
load_time_re = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
sample_time_re = re.compile(r"sample time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
prompt_eval_time_re = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
eval_time_re = re.compile(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second\)")
total_time_re = re.compile(r"total time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")

# Function to extract llama.cpp timing details from stderr output
def extract_timings(stderr_output):
    load_time_match = load_time_re.search(stderr_output)
    load_time = float(load_time_match.group(1)) if load_time_match else None

    sample_time_match = sample_time_re.search(stderr_output)
    sample_time = float(sample_time_match.group(1)) if sample_time_match else None
    sample_runs = int(sample_time_match.group(2)) if sample_time_match else None
    sample_throughput = float(sample_time_match.group(4)) if sample_time_match else None

    prompt_eval_match = prompt_eval_time_re.search(stderr_output)
    prefill_time = float(prompt_eval_match.group(1)) if prompt_eval_match else None
    prefill_tokens = int(prompt_eval_match.group(2)) if prompt_eval_match else None
    prefill_throughput = float(prompt_eval_match.group(4)) if prompt_eval_match else None

    eval_time_match = eval_time_re.search(stderr_output)
    decode_time = float(eval_time_match.group(1)) if eval_time_match else None
    decode_runs = int(eval_time_match.group(2)) if eval_time_match else None
    decode_throughput = float(eval_time_match.group(4)) if eval_time_match else None

    total_time_match = total_time_re.search(stderr_output)
    total_time = float(total_time_match.group(1)) if total_time_match else None
    total_tokens = int(total_time_match.group(2)) if total_time_match else None

    return {
        'load_time': load_time,
        'sample_time': sample_time,
        'sample_runs': sample_runs,
        'sample_throughput': sample_throughput,
        'prefill_time': prefill_time,
        'prefill_tokens': prefill_tokens,
        'prefill_throughput': prefill_throughput,
        'decode_time': decode_time,
        'decode_runs': decode_runs,
        'decode_throughput': decode_throughput,
        'total_time': total_time,
        'total_tokens': total_tokens
    }

# Function to capture stdout and stderr
def capture_output(func, *args, **kwargs):
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        result = func(*args, **kwargs)
        output = sys.stdout.getvalue()
        error_output = sys.stderr.getvalue()
        #print("stderr_output", error_output)
    finally:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        
    return result, output, error_output

# Updated evaluate_conversations function to write all the timing details to CSV
def evaluate_conversations(messages, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([
            'Message ID', 'LLM Response', 'Inference Time (seconds)', 'Throughput (tokens/second)', 
            'Model Loading Time (seconds)', 'Load Time (ms)', 'Sample Time (ms)', 'Sample Runs', 
            'Sample Throughput (tokens/second)', 'Prefill Time (ms)', 'Prefill Tokens', 
            'Prefill Throughput (tokens/second)', 'Decode Time (ms)', 'Decode Runs', 
            'Decode Throughput (tokens/second)', 'Total Time (ms)', 'Total Tokens', 'Timestamp'
        ])

        for message_index, message in enumerate(messages):
            if isinstance(message, str):
                print("msg ", message_index)
                start_time = time.time()
                llama_output, output, stderr_output = capture_output(lambda: llm(message, max_tokens=500))
                end_time = time.time()
                print("stderr_output",stderr_output)
                
                inference_time = end_time - start_time
                response_text = llama_output['choices'][0]['text'].replace('\n', ' ').strip()

                tokens_generated = len(message.split()) + len(response_text.split())
                throughput = tokens_generated / inference_time if inference_time > 0 else 0
                
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                
                # Extract timings from stderr_output
                timings = extract_timings(stderr_output)
                #print("timings",timings)
                csv_writer.writerow([
                    f"{message_index}", 
                    response_text, 
                    inference_time, 
                    throughput, 
                    model_loading_time,
                    timings['load_time'] if timings['load_time'] is not None else "N/A",
                    timings['sample_time'] if timings['sample_time'] is not None else "N/A",
                    timings['sample_runs'] if timings['sample_runs'] is not None else "N/A",
                    timings['sample_throughput'] if timings['sample_throughput'] is not None else "N/A",
                    timings['prefill_time'] if timings['prefill_time'] is not None else "N/A",
                    timings['prefill_tokens'] if timings['prefill_tokens'] is not None else "N/A",
                    timings['prefill_throughput'] if timings['prefill_throughput'] is not None else "N/A",
                    timings['decode_time'] if timings['decode_time'] is not None else "N/A",
                    timings['decode_runs'] if timings['decode_runs'] is not None else "N/A",
                    timings['decode_throughput'] if timings['decode_throughput'] is not None else "N/A",
                    timings['total_time'] if timings['total_time'] is not None else "N/A",
                    timings['total_tokens'] if timings['total_tokens'] is not None else "N/A",
                    timestamp
                ])
            else:
                print(f"Skipping non-string message: {message}")

# Function to run the experiment 3 times
def run_experiments_3_times(messages, base_output_file):
    for i in range(5):
        output_file = f'{base_output_file}_run_{i+1}.csv'
        print(f'Running experiment {i+1}, saving results to {output_file}')
        evaluate_conversations(messages, output_file)
        time.sleep(100)  # Optional delay between runs

# Load messages from conversations.json
with open('./prompt/conversations.json', 'r') as f:
    conversations = json.load(f)

# Flatten the conversations into a single list of messages
flat_messages = [message for conversation in conversations for message in conversation]

# Run the experiments 3 times for the conversations.json
run_experiments_3_times(flat_messages, './results/inference_metrics_conversations')

print ("experiments completed!")
# Add a prompt to wait for the user to confirm before exiting
input("Copy the results and press Enter to finish...")
while True:
    pass
