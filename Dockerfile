# Matches the local dev environment's Python version (3.13) rather than
# a generic LTS tag, since every dependency in requirements.txt was
# actually verified against 3.13 locally - no point introducing an
# untested interpreter version at the one point (deployment) where a
# mismatch would be hardest to debug.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# chroma_db/ and data/processed/ ship as real, already-built content, not
# rebuilt at image-build time: a full embedding build takes ~90 minutes
# under Voyage's free-tier 3-requests/minute throttle, which exceeds any
# realistic build timeout and would re-spend the same API budget on
# every deploy for no reason - the vectors don't change between deploys.
COPY . .

# Hugging Face Spaces' Docker SDK expects the app to listen on 7860.
EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
