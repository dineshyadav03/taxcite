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

Orchestration is LangGraph (a StateGraph over agent/tools nodes with
conditional routing) rather than the hand-rolled while-loop this pattern
started with - the standard way this control-flow shape gets built in
practice, and worth demonstrating directly rather than only ever hand-
rolling it. Deliberately NOT also switching the model-calling layer to
LangChain's own primitives (ChatGroq, bind_tools): generate.llm() already
carries real, working retry-with-backoff for Groq's rate limits, JSON-
mode salvage-parsing, and (as of this session) per-request token
accounting that src/observability.py's cost tracking depends on -
swapping the actual LLM call to a separate LangChain wrapper would
silently stop tokens from being counted, since that accounting hangs off
generate.llm() specifically. LangGraph owns the control flow; every
model call still goes through the same primitive every other pattern
uses.
"""
import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph

import graph
from generate import DISTANCE_REFUSAL_THRESHOLD, REFUSAL_MESSAGE, build_user_prompt, llm
from retrieve import exact_section_lookup, fetch_section, search, vector_search

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


def _confidently_grounded(question):
    """Same two signals every other pattern's refusal gate is built from
    - the question's own text, checked once, deterministically - NOT
    anything the agent's tool loop happened to find along the way.

    Two earlier versions of this check both failed live testing before
    landing here, worth recording since the failure modes are subtle:

    1. First attempt checked the full `gathered` list (every chunk from
    every tool call). Broke on the cookie question: a "baking recipe"
    search surfaced an unrelated industrial-schedule cross-reference
    (Section 9, food-processing tax deductions), the agent got curious
    and called get_section on it, and its 11 real chunks (distance=0.0
    by construction - get_section only returns real, existing sections)
    made the whole list look confidently grounded even though nothing
    found was ever actually about cookies. A section existing doesn't
    mean it's relevant when the AGENT, not the question, decided to
    fetch it.

    2. Second attempt scoped the check to search_hits only, plus an
    exact_section_lookup(question) bypass for direct section-number
    questions (correctly fixing case 1 - verified live on "what does
    section 292 say", where the agent smartly skips search() entirely
    and goes straight to get_section). But it still broke on the SAME
    cookie question a different way: Agentic tries several rephrasings
    across its exploration ("baking recipe", "chocolate chip cookies
    recipe", ...), and by pure food-adjacent vocabulary overlap, one
    rephrasing ("baking recipe") scored distance 1.0 against that same
    Section 9 schedule - a false positive. More exploratory attempts
    means more chances to stumble into a spurious low-distance match
    somewhere in a 2,400-chunk corpus; a permissive "any search hit
    under threshold" check is more exploitable the more searches run.

    This version sidesteps both failure modes: exact_section_lookup for
    the direct-citation case, else ONE raw vector_search on the
    question's own literal text - the same deterministic signal
    DISTANCE_REFUSAL_THRESHOLD was originally calibrated against, and
    the same approach HyDE's own refusal-gate fix uses. The agent's tool
    loop can still explore freely to gather content for the answer;
    only the refusal decision itself is grounded this way."""
    if exact_section_lookup(question):
        return True
    raw_hits = vector_search(question, top_k=1)
    return bool(raw_hits) and raw_hits[0]["distance"] <= DISTANCE_REFUSAL_THRESHOLD


class _AgentState(TypedDict):
    question: str
    transcript: str
    gathered: list
    # Annotated + operator.add makes LangGraph append each node's
    # returned list to the running one automatically, instead of
    # overwriting it - the natural fit for a trace that should show
    # every step across repeated agent/tools cycles, not just the last.
    steps: Annotated[list, operator.add]
    step_count: int
    action: dict
    outcome: dict  # set by finish_node/forced_finish_node; answer() reads this


def _agent_node(state):
    try:
        # Raised from 500 after the Groq model swap (see generate.py's
        # GROQ_MODEL comment) - the new model reasons internally before
        # writing the JSON action, and that reasoning counts against this
        # same budget, called once per tool-loop step (up to 5 times).
        action = llm(state["transcript"], system_prompt=_AGENT_SYSTEM, json_mode=True, max_tokens=1200)
    except ValueError as e:
        return {"action": {"tool": "__error__"}, "steps": [{"error": str(e)[:200]}]}
    return {"action": action, "step_count": state["step_count"] + 1}


def _tools_node(state):
    action = state["action"]
    gathered = state["gathered"]
    observation = _run_tool(action, gathered)
    transcript = state["transcript"] + (
        f"\n\nAction taken: {json.dumps(action)}\nObservation: {json.dumps(observation)[:2000]}"
        f"\n\nNext action (step {state['step_count'] + 1}/{_MAX_STEPS}, finish when ready):"
    )
    return {
        "gathered": gathered,
        "transcript": transcript,
        "steps": [{"action": action, "observation_preview": json.dumps(observation)[:300]}],
    }


def _finish_node(state):
    action = state["action"]
    answer_text = str(action.get("answer", "")).strip()
    if not answer_text:
        return _forced_finish_node(state)
    return {"steps": [{"action": "finish"}], "outcome": {"answer_text": answer_text, "forced": False}}


def _forced_finish_node(state):
    # Step budget exhausted, a malformed action, or an empty finish
    # answer - force a decision from whatever was gathered rather than
    # burning more calls. The actual grounded/refused decision happens
    # once in answer() after the graph returns (see _confidently_grounded's
    # docstring for why it's independent of anything the loop gathered);
    # this node only decides whether there's enough material to even
    # attempt a forced answer.
    return {"steps": [{"forced_finish": True}], "outcome": {"answer_text": None, "forced": True}}


def _route_after_agent(state):
    tool = state["action"].get("tool")
    if tool == "finish":
        return "finish"
    if tool == "__error__" or state["step_count"] > _MAX_STEPS:
        return "forced_finish"
    return "tools"


def _route_after_tools(state):
    return "forced_finish" if state["step_count"] >= _MAX_STEPS else "agent"


def _build_graph():
    g = StateGraph(_AgentState)
    g.add_node("agent", _agent_node)
    g.add_node("tools", _tools_node)
    g.add_node("finish", _finish_node)
    g.add_node("forced_finish", _forced_finish_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", _route_after_agent, {"finish": "finish", "tools": "tools", "forced_finish": "forced_finish"})
    g.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", "forced_finish": "forced_finish"})
    g.add_edge("finish", END)
    g.add_edge("forced_finish", END)
    return g.compile()


_GRAPH = _build_graph()


def answer(question, session_id=None):
    grounded = _confidently_grounded(question)

    final_state = _GRAPH.invoke(
        {
            "question": question,
            "transcript": f"Question: {question}",
            "gathered": [],
            "steps": [],
            "step_count": 0,
            "action": {},
            "outcome": {},
        },
        config={"recursion_limit": _MAX_STEPS * 2 + 4},
    )

    trace = {"steps": final_state["steps"], "confidently_grounded": grounded}
    gathered = final_state["gathered"]
    outcome = final_state["outcome"]

    # The system prompt asks the model to self-report when the corpus
    # doesn't answer the question, but nothing enforced that - verified
    # live: a chocolate-chip-cookie question got an honest "I don't have
    # any information..." answer, yet `refused` was unconditionally False
    # regardless of what the text actually said, the same gap HyDE had
    # before its own fix. Checked here against the actual retrieval
    # evidence instead of trusting the model's own free-text framing.
    if not outcome["forced"]:
        if not grounded:
            return {"answer": REFUSAL_MESSAGE, "chunks": gathered[:8], "refused": True, "trace": trace}
        return {"answer": outcome["answer_text"], "chunks": gathered[:8], "refused": False, "trace": trace}

    if gathered and grounded:
        trace["forced_finish"] = True
        answer_text = llm(build_user_prompt(question, gathered[:6]))
        return {"answer": answer_text, "chunks": gathered[:6], "refused": False, "trace": trace}
    return {"answer": REFUSAL_MESSAGE, "chunks": gathered[:6], "refused": True, "trace": trace}
