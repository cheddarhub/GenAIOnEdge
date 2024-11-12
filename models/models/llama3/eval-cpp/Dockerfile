# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app
COPY model/ /app/model
COPY results/ /app/results
COPY prompt/ /app/prompt
# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    llama-cpp-python==0.2.90

COPY llama3-eval-cpp.py /app
CMD ["python", "llama3-eval-cpp.py"]

