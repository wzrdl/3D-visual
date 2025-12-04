FROM python:3.11-slim

# Install basic build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project code into the image
COPY . /app

# Install backend and shared Python dependencies
RUN pip install --no-cache-dir -r server/requirements.txt -r requirements.txt

# Cloud Run uses the PORT environment variable to define the listening port
ENV PORT=8080

# Expose port (for documentation; Cloud Run handles networking)
EXPOSE 8080

# Start FastAPI backend
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8080"]




