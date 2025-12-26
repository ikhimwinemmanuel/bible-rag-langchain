# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set a working directory inside the container
WORKDIR /app

# 3. Copy dependency file first (for caching)
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port Gradio runs on
EXPOSE 7860

# 7. Start the application
CMD ["python", "app.py"]
