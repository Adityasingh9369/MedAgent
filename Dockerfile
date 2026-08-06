FROM python:3.12-slim

WORKDIR /app

# System dependencies needed for psycopg2, PyMuPDF, and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --timeout=180 --retries=5 -r requirements.txt

# spaCy model — same wheel URL used locally
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# httpx pin, same as local (avoids the openai client 'proxies' TypeError)
RUN pip install --no-cache-dir httpx==0.27.0

# Now copy the actual application code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]