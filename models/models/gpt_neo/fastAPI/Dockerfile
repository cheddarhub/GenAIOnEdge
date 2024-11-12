    # Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch
Run pip install transformers
Run pip install "fastapi[standard]"
Copy gptneo-fast.py /app
Copy model/ /app/model
CMD ["python", "-m", "fastapi", "run", "gptneo-fast.py"]

