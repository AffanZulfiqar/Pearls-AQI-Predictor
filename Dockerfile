FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for Hopsworks and ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on (Hugging Face expects port 7860 by default)
EXPOSE 7860

# Command to run the Flask API (uses the PORT environment variable provided by Railway)
CMD flask --app src.inference.api run --host=0.0.0.0 --port=${PORT:-5000}
