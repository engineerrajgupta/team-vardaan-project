# Dockerfile

# 1. Start with a lightweight official Python base image
FROM python:3.11-slim-buster

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install Tesseract OCR engine and then Python packages
# This is the build command, now inside our own environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code into the container
COPY . .

# 6. Define the command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
