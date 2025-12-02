# Use official Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY req.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

