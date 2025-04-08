FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files first
COPY models/ ./models/

# Copy the rest of the application
COPY . .

EXPOSE 10000

# Use 2 workers, increase timeout, and use gthread worker class for better handling of ML models
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "--worker-class", "gthread", "--threads", "4", "app:app"] 