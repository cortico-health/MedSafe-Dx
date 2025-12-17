FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Add app to PYTHONPATH
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "-m", "inference.run_inference"]

