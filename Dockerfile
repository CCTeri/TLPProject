# Use Python runtime as a base image
FROM python:3.11-slim

# Install the OpenMP runtime (for LightGBM to work in GCP)
# libgomp1 is required for LightGBM to run in linux
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (to optimize caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Setting port
EXPOSE 8080

# Command to run the application
CMD ["python", "server.py"]
