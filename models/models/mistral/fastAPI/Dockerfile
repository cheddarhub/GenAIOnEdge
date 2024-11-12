# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version) and other dependencies
RUN pip install --no-cache-dir ctransformers --no-binary ctransformers
RUN pip install --no-cache-dir llama-cpp-python
RUN pip install --no-cache-dir "fastapi[standard]"
COPY mistralgguf-fast.py /app
COPY ./model /app/model
VOLUME ["/data"]
CMD ["python", "-m", "fastapi", "run", "mistralgguf-fast.py"]

