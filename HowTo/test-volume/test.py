# app.py
import os
import time  # Import time module
from datetime import datetime

# Directory where the file will be created (mounted in Kubernetes)
directory = "/data"

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# File path inside the mounted directory
file_path = os.path.join(directory, "output.txt")

# Write the current timestamp to the file
with open(file_path, "a") as file:
    file.write(f"Timestamp: {datetime.now()}\n")

print(f"Written to file: {file_path}")

while True:  # Infinite loop to run the code continuously
    with open(file_path, "a") as file:
        file.write(f"Timestamp: {datetime.now()}\n")
    
    print(f"Written to file: {file_path}")
    time.sleep(180)  # Sleep for 180 seconds (3 minutes)
