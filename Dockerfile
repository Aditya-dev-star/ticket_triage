FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY README.md .
COPY openenv.yaml .
COPY __init__.py .
COPY models.py .
COPY client.py .
COPY server/ ./server/

# Install dependencies including the local package and openai for the client
RUN pip install . openai requests uvicorn fastapi openenv-core

# Expose the API port
EXPOSE 8000

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
