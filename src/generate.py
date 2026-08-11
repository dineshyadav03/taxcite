"""LLM access + the baseline cited-answer pipeline (Simple RAG).

Generation backend: Groq (free tier) when GROQ_API_KEY is set, else local
Ollama - same two-backend choice as the sibling ai-incident-rag project,
for the same reason (Groq gets generation off local CPU, which matters on
a machine under real background load).

llm() is the shared primitive every pattern in src/patterns/ builds on:
arbitrary system prompt, optional JSON mode, retry-with-backoff against
Groq's free-tier rate limits. answer_question() is the original Simple
RAG pipeline, kept intact because eval/run_eval.py scores against it.
"""
import json
import os
import re
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

# Calibrated against real eval failures, not guessed: 0.9 (the original
# value, picked before switching to Voyage AI) caused 3/20 golden-set
# questions to refuse even though retrieval had found the right section
# (retrieval_hit_rate was 1.00 while citation/pass rate was lower) -
# Voyage's distance scale for genuinely relevant-but-not-lexically-obvious
# matches runs ~0.9-1.15, while verified off-corpus queries (chocolate
# chip cookies, GST) score ~1.2-1.4. 1.15 sits in that gap.
DISTANCE_REFUSAL_THRESHOLD = float(os.environ.get("DISTANCE_REFUSAL_THRESHOLD", "1.15"))

REFUSAL_MESSAGE = (
    "I don't have a well-matched section in the indexed Acts for this question. "
    "This corpus covers the Income-tax Act, 2025 and the Income-tax Act, 1961 only."
)

SYSTEM_PROMPT = """You are a research assistant answering questions about Indian income tax law, grounded ONLY in the retrieved Act sections provided below. Rules:
- Answer using only the retrieved text. Do not use outside knowledge of tax law, even if you believe you know the answer.
- Every claim must be attributable to a specific retrieved section. Cite the real Act title and section number inline in parentheses, e.g. "(Income-tax Act, 1961, Section 139)" - use the actual values from the retrieved sections, never the literal words "Act title" or "Section N".
- If the retrieved sections don't actually answer the question, say so plainly instead of guessing.
- Be precise about which Act (2025 or 1961) a provision is from - they are different laws with different section numbering."""

_groq_client = None


def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def llm(user_prompt, system_prompt=SYSTEM_PROMPT, json_mode=False, max_tokens=None, temperature=0, model=None):
    """Single LLM call with retry-with-backoff. When json_mode=True, asks
    Groq for a JSON object response and parses it (returns a dict/list, or
    raises ValueError if parsing fails after a salvage attempt). model
    overrides GROQ_MODEL for this call only - used sparingly (e.g. the
    Correspondence RAG build script upgrades to llama-3.3-70b-versatile
    for its verification calls, since that judgment is more nuanced than
    the everyday 8B model reliably gets right - verified live: the 8B
    model kept mislabeling "one exemption among fifty" as a "strong"
    match to the whole exemptions section)."""
    max_tokens = max_tokens or OLLAMA_NUM_PREDICT
    if GROQ_API_KEY:
        from groq import APIConnectionError, BadRequestError, RateLimitError

        client = _get_groq()
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(GROQ_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model or GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                text = response.choices[0].message.content
                break
            except (RateLimitError, APIConnectionError):
                # APIConnectionError (covers APITimeoutError) verified live
                # during eval expansion - two separate patterns hit a
                # transient network timeout back-to-back and crashed
                # outright instead of retrying, since only RateLimitError
                # was being caught here. A connection blip deserves the
                # same backoff-and-retry treatment as a rate limit, not a
                # hard failure on the first bad network moment.
                if attempt == GROQ_MAX_RETRIES - 1:
                    raise
                time.sleep(2**attempt)
            except BadRequestError as e:
                # Groq's JSON-mode validator hard-errors (400,
                # json_validate_failed - usually a truncated JSON object
                # from hitting max_tokens mid-generation) rather than
                # returning salvageable text. Verified live: self_rag's
                # critique call crashed the whole pattern on this before
                # being folded into the same ValueError every caller
                # already handles from parse_json's own salvage failure.
                if json_mode and "json" in str(e).lower():
                    raise ValueError(f"Groq JSON mode failed: {e}") from e
                raise
    else:
        import ollama

        client = ollama.Client()
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_ctx": OLLAMA_NUM_CTX, "num_predict": max_tokens},
            format="json" if json_mode else "",
        )
        text = response["message"]["content"]

    if not json_mode:
        return text
    return parse_json(text)


def parse_json(text):
    """Parse LLM JSON output, salvaging the largest {...} block if the
    model wrapped it in prose (small models do this even in JSON mode)."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"LLM did not return parseable JSON: {text[:200]!r}")


# Groq's free tier caps llama-3.1-8b-instant at a hard 6,000-token
# tokens-per-minute ceiling - a single oversized prompt is rejected
# outright (413), not queued or retried. Verified live twice: once in
# Correspondence RAG's own prompt construction (a per-section chunk cap
# fixed it there), and again here in graph_rag.py after hybrid retrieval
# (see retrieve.py) started filling every pattern's chunk slots more
# reliably - more real recall means bigger prompts more often, which is
# exactly the failure mode a shared budget in the one function every
# pattern's final generation call goes through can prevent centrally,
# instead of re-discovering and re-fixing it pattern by pattern.
# ~4 characters/token is a conservative average for English legal prose
# (real tiktoken counts run a bit better than this on purpose, for
# margin); 16,000 chars leaves headroom for the system prompt, the
# question, and the model's own output tokens within the same allowance.
_MAX_CONTEXT_CHARS = 16000


def build_user_prompt(question, chunks, history=None):
    context_blocks = []
    budget = _MAX_CONTEXT_CHARS
    for c in chunks:
        if budget <= 0:
            break
        m = c["metadata"]
        label = m.get("act_title", "Unknown")
        if m.get("section"):
            label += f", Section {m['section']}"
        if m.get("modality") == "table":
            label += " (table)"
        block = f"[{label}]\n{c['text']}"
        if len(block) > budget:
            block = block[:budget] + "..."
        context_blocks.append(block)
        budget -= len(block)
    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"Retrieved sections:\n\n{context}\n\n---\n\n"
    if history:
        prompt += f"Conversation so far:\n{history}\n\n---\n\n"
    prompt += f"Question: {question}"
    return prompt


def _generate(user_prompt):
    return llm(user_prompt)


def answer_question(question, top_k=5, embedding=None):
    """The original Simple RAG pipeline (pattern 1). eval/run_eval.py
    scores against this - keep its contract stable. embedding lets a
    caller (e.g. Jury RAG) reuse an already-computed query vector."""
    chunks = search(question, top_k=top_k, embedding=embedding)

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
