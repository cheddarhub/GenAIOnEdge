# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install PyTorch (CPU version)
RUN pip install "fastapi[standard]"
RUN pip install requests
COPY main.py /app
CMD ["python", "-m", "fastapi", "run", "main.py"]

