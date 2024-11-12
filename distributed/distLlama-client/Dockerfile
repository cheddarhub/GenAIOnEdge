# Use the official Python image from Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
RUN pip install requests

# Copy the rest of the application files into the container
COPY prompts/ /app/prompts/
COPY results/ /app/results
COPY distLlama-simple.py /app

# Set the default command to run your Python script
CMD ["python", "distLlama-simple.py"]
