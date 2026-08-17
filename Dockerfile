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

# 7860 is only the default, not a hardcoded requirement - Hugging Face
# Spaces' Docker SDK specifically expects 7860, but Render (and most
# other platforms) inject their own port via a $PORT env var at
# container start and route traffic there instead, whatever it is.
# Reading $PORT with a 7860 fallback means the same image runs correctly
# on either kind of platform, and on a plain local `docker run` with no
# $PORT set at all - one image, not a per-platform variant.
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-7860}"]
