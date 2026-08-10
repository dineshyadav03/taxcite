"""Pattern 8 - Agentic RAG: the LLM drives retrieval as a tool loop.

Every other pattern hardcodes its retrieval strategy; here the model
chooses each step. Four tools, JSON actions, max 5 steps:

  search(query)                - vector+exact search, compact results
  get_section(act_id, section) - full text of one exact section
  get_references(act_id, section) - what it cites / what cites it (graph)
  finish(answer)               - final cited answer

Worth being honest about scale: with an 8B model as the driver, this is
the pattern most likely to wander - the trace (full action log) exists
precisely so a reviewer can see *how* it navigated, including when a
smaller model takes a redundant step. The forced-finish fallback caps
cost: after 5 steps it must answer from whatever it has gathered.
"""
import json

import graph
from generate import REFUSAL_MESSAGE, llm
from retrieve import fetch_section, search

_MAX_STEPS = 5

_AGENT_SYSTEM = """You are a legal research agent for Indian income tax law with retrieval tools. Each turn, reply with exactly one JSON action:
{"tool": "search", "query": "..."} - semantic + exact search over both Acts
{"tool": "get_section", "act_id": "itact2025" or "itact1961", "section": "..."} - full text of a section
{"tool": "get_references", "act_id": "...", "section": "..."} - cross-references of a section
{"tool": "finish", "answer": "..."} - final answer, citing (Act title, Section N) for every claim

Rules: ground every claim in text you actually retrieved this conversation; if the corpus doesn't answer the question, finish with an honest statement of that; prefer finishing as soon as you have enough - each tool call has real cost."""


def _run_tool(action, gathered):
    tool = action.get("tool")
    if tool == "search":
        hits = search(str(action.get("query", "")), top_k=4)
        gathered.extend(hits)
        return [
            {"act": c["metadata"].get("act_id"), "section": c["metadata"].get("section"), "distance": round(c["distance"], 3), "preview": c["text"][:200]}
            for c in hits
        ]
    if tool == "get_section":
        hits = fetch_section(str(action.get("act_id", "")), str(action.get("section", "")))
        gathered.extend(hits)
        if not hits:
            return {"error": "no such section in that Act"}
        return {"act": action.get("act_id"), "section": action.get("section"), "text": " ".join(c["text"] for c in hits)[:2500]}
    if tool == "get_references":
        act_id = str(action.get("act_id", ""))
        section = str(action.get("section", ""))
        return {
            "cites": graph.references_of(act_id, section)[:15],
            "cited_by": graph.referenced_by(act_id, section)[:15],
        }
    return {"error": f"unknown tool {tool!r}"}


def answer(question, session_id=None):
    trace = {"steps": []}
    gathered = []
    transcript = f"Question: {question}"

    for step in range(_MAX_STEPS):
        try:
            action = llm(transcript, system_prompt=_AGENT_SYSTEM, json_mode=True, max_tokens=500)
        except ValueError as e:
            trace["steps"].append({"error": str(e)[:200]})
            break

        if action.get("tool") == "finish":
            trace["steps"].append({"action": "finish"})
            answer_text = str(action.get("answer", "")).strip()
            if not answer_text:
                break
            return {"answer": answer_text, "chunks": gathered[:8], "refused": False, "trace": trace}

        observation = _run_tool(action, gathered)
        trace["steps"].append({"action": action, "observation_preview": json.dumps(observation)[:300]})
        transcript += (
            f"\n\nAction taken: {json.dumps(action)}\nObservation: {json.dumps(observation)[:2000]}"
            f"\n\nNext action (step {step + 2}/{_MAX_STEPS}, finish when ready):"
        )

    # step budget exhausted or malformed action - force an answer from
    # whatever was gathered rather than burning more calls
    if gathered:
        from generate import build_user_prompt

        trace["forced_finish"] = True
        answer_text = llm(build_user_prompt(question, gathered[:6]))
        return {"answer": answer_text, "chunks": gathered[:6], "refused": False, "trace": trace}
    return {"answer": REFUSAL_MESSAGE, "chunks": [], "refused": True, "trace": trace}
