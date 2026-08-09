"""Build a cited prompt from retrieved sections and generate an answer.

v1 generation backend: Groq (free tier) when GROQ_API_KEY is set, else
local Ollama - same two-backend choice as the sibling ai-incident-rag
project, for the same reason (Groq gets generation off local CPU, which
matters on a machine under real background load).
"""
import os
import time

from dotenv import load_dotenv

from retrieve import search

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "6144"))
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "500"))
GROQ_MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "3"))

# Calibrated empirically (see scripts/calibrate_threshold.py output): the
# top match's cosine distance for genuinely relevant queries vs. off-corpus
# ones. Refusing beyond this avoids confidently answering from an
# irrelevant section - the exact failure mode this project exists to avoid.
DISTANCE_REFUSAL_THRESHOLD = float(os.environ.get("DISTANCE_REFUSAL_THRESHOLD", "0.9"))

REFUSAL_MESSAGE = (
    "I don't have a well-matched section in the indexed Acts for this question. "
    "This corpus covers the Income-tax Act, 2025 and the Income-tax Act, 1961 only."
)

SYSTEM_PROMPT = """You are a research assistant answering questions about Indian income tax law, grounded ONLY in the retrieved Act sections provided below. Rules:
- Answer using only the retrieved text. Do not use outside knowledge of tax law, even if you believe you know the answer.
- Every claim must be attributable to a specific retrieved section. Cite as (Act title, Section N) inline.
- If the retrieved sections don't actually answer the question, say so plainly instead of guessing.
- Be precise about which Act (2025 or 1961) a provision is from - they are different laws with different section numbering."""


def build_user_prompt(question, chunks):
    context_blocks = []
    for c in chunks:
        m = c["metadata"]
        context_blocks.append(f"[{m['act_title']}, Section {m['section']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"Retrieved sections:\n\n{context}\n\n---\n\nQuestion: {question}"


def _generate_groq(user_prompt):
    from groq import RateLimitError, Groq

    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(GROQ_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=OLLAMA_NUM_PREDICT,
            )
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == GROQ_MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)


def _generate_ollama(user_prompt):
    import ollama

    client = ollama.Client()
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT},
    )
    return response["message"]["content"]


def _generate(user_prompt):
    if GROQ_API_KEY:
        return _generate_groq(user_prompt)
    return _generate_ollama(user_prompt)


def answer_question(question, top_k=5):
    chunks = search(question, top_k=top_k)

    if not chunks or chunks[0]["distance"] > DISTANCE_REFUSAL_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "chunks": chunks, "refused": True}

    user_prompt = build_user_prompt(question, chunks)
    answer_text = _generate(user_prompt)
    return {"answer": answer_text, "chunks": chunks, "refused": False}


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the tax treatment of house rent allowance?"
    result = answer_question(question)
    print(f"Q: {question}\n\n{result['answer']}\n")
    if not result["refused"]:
        print("Sources:")
        for c in result["chunks"]:
            m = c["metadata"]
            print(f"  - {m['act_title']}, Section {m['section']} (distance={c['distance']:.3f})")
