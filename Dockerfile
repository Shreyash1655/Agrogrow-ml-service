# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your machine learning code
COPY . .

# Expose port 8000 (the default for FastAPI)
EXPOSE 8000

# Start the FastAPI server using Uvicorn
# Note: If your main python file is named 'app.py', change 'main:app' to 'app:app'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]