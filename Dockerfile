FROM python:3.13-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/evopt ./evopt

# Expose the port
EXPOSE 7050

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7050", "--workers", "4", "--max-requests", "32", "evopt.app:app"]
