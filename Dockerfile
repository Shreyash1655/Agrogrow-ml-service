# ✅ FIXED: Upgraded from 3.9 to 3.11 to support your package requirements
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your machine learning code
COPY . .

# Expose port 8000 (default for FastAPI)
EXPOSE 8000

# Start the FastAPI server
# Make sure your main file is named 'main.py'. If it's 'app.py', change to 'app:app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]