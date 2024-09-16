# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements (dependencies) into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Streamlit runs on (default 8501)
EXPOSE 8501

# Command to run Jupyter Lab (optional)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
