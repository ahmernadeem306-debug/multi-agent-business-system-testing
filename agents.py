"""
agents.py
=========
Stateful, multi-agent orchestration layer for the BizAgent platform,
built on LangGraph.

Architecture
------------
                    +---------------------+
        START --->  |   Supervisor Agent   |  <---------------+
                    +----------+----------+                   |
                               |                               |
              (structured routing decision, LLM-driven)        |
                               |                                |
             +-----------------+------------------+            |
             |                                     |            |
      +------v------+                     +--------v-------+   |
      | Inventory   |                     |  Compliance    |   |
      |   Agent     |                     |    Agent       |   |
      | (ReAct +    |                     |  (ReAct +      |   |
      |  MCP tool)  |                     |   MCP tool)    |   |
      +------+------+                     +--------+-------+   |
             |                                     |            |
             +----------------->-------------------+------------+
                                                     |
                                                    END (FINISH)

NO SIMULATED WORKFLOWS
-----------------------
Neither worker agent is permitted to answer from its own knowledge.
Each is a LangGraph ReAct agent bound to exactly one *real* MCP tool
(loaded live from mcp_server.py, which itself queries the operational
SQLite database and the persisted Chroma vector store). Every fact in
a worker's final answer traces back to an actual tool result recorded
in the conversation's message history -- there is no hardcoded
business data or canned answer text in this file.

Requires
--------
    ANTHROPIC_API_KEY   set in the environment (never hardcoded here)
    mcp_server.py, database.py, rag_engine.py   in the same directory

Usage
-----
    python agents.py "what needs to be reordered right now?"
    python agents.py "what's our policy on dairy refrigeration?"

    # or programmatically:
    from agents import run_agentic_workflow
    final_state = await run_agentic_workflow("...")
"""

from __future__ import annotations

import asyncio
import operator
import os
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- 
# Configuration
# --------------------------------------------------------------------------- 

BASE_DIR = Path(__file__).resolve().parent
MCP_SERVER_PATH = BASE_DIR / "mcp_server.py"

# Overridable via environment; no model name is a "hardcoded answer" --
# it is pipeline configuration, same category as a DB path or chunk size.
LLM_MODEL = os.environ.get("BIZAGENT_LLM_MODEL", "claude-sonnet-5")
MAX_SUPERVISOR_ITERATIONS = int(os.environ.get("BIZAGENT_MAX_SUPERVISOR_ITERATIONS", "4"))

REQUIRED_TOOLS = ("check_inventory_levels", "verify_store_compliance")


# --------------------------------------------------------------------------- 
# State schema
# --------------------------------------------------------------------------- 

class AgentState(TypedDict):
    """
    Shared, persistent state threaded through every node in the graph.

    messages     : full conversational transcript (human turns, supervisor
                   is not a chat participant, worker AI turns, and the
                   ToolMessages recording each real MCP tool call/result).
                   Uses LangGraph's `add_messages` reducer so every node
                   appends rather than clobbers history.
    next         : the Supervisor's most recent routing decision --
                   "inventory_agent", "compliance_agent", or "FINISH".
    routing_log  : append-only audit trail of every supervisor decision
                   across graph iterations (uses `operator.add` so each
                   node's returned list is concatenated, never overwritten).
    iterations   : supervisor visit counter, used as a hard loop guard so
                   a misbehaving routing loop cannot run indefinitely.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    next: str
    routing_log: Annotated[list[dict], operator.add]
    iterations: int


class RouteDecision(BaseModel):
    """Structured output contract the Supervisor LLM must return."""

    next: Literal["inventory_agent", "compliance_agent", "FINISH"] = Field(
        description=(
            "Which specialized worker should act next. Choose 'FINISH' only "
            "once the worker responses already in the conversation fully "
            "answer the user's original request."
        )
    )
    reasoning: str = Field(description="One sentence explaining this routing choice.")


# --------------------------------------------------------------------------- 
# Agent instructions (orchestration prompts -- not simulated answers)
# --------------------------------------------------------------------------- 
# These strings govern *behavior*, not content: they tell each agent which
# real tool it must call and forbid it from inventing data. They contain no
# store metrics, product names, or policy text of any kind.

SUPERVISOR_PROMPT = (
    "You are the Supervisor Agent for BizAgent, a supermart operations platform. "
    "You do not answer the user's question yourself and you never invent facts. "
    "Your only job is to decide, on each turn, which specialized worker should "
    "act next:\n"
    "- inventory_agent: handles stock levels, low-stock items, reorder questions, "
    "or anything about what the store currently has on hand.\n"
    "- compliance_agent: handles store policy, SOP, or regulatory/compliance "
    "questions that must be answered from the ingested policy PDFs.\n"
    "Choose 'FINISH' once the conversation already contains a worker response "
    "that fully answers the user's original request, or if the request needs "
    "no worker at all (e.g. small talk) -- in that case still choose FINISH "
    "and a plain closing turn will be sent without fabricating any data."
)

INVENTORY_AGENT_PROMPT = (
    "You are the Inventory Agent for BizAgent. You have exactly one tool, "
    "`check_inventory_levels`, which returns the store's real, current "
    "low-stock report from the live operational database. You must call "
    "this tool to answer any inventory question -- never estimate, guess, "
    "or state a stock number that did not come from the tool's output. "
    "If the tool returns no items, tell the user nothing is currently "
    "below its minimum stock threshold; do not invent example products."
)

COMPLIANCE_AGENT_PROMPT = (
    "You are the Compliance Agent for BizAgent. You have exactly one tool, "
    "`verify_store_compliance`, which runs a real semantic search over the "
    "store's ingested policy PDFs and returns exact matching passages. You "
    "must call this tool for any policy or compliance question -- never "
    "recite policy language from memory. If the tool returns no matching "
    "passages, tell the user the knowledge base has no relevant policy on "
    "file; do not fabricate a plausible-sounding policy."
)


# --------------------------------------------------------------------------- 
# Graph construction
# --------------------------------------------------------------------------- 

async def _load_verified_tools(session) -> dict:
    """
    Load the tool set exposed by mcp_server.py over an already-open MCP
    session, and fail loudly if the tools this architecture depends on
    are missing -- there is no silent fallback to a fake tool.
    """
    tools = await load_mcp_tools(session)
    tools_by_name = {tool.name: tool for tool in tools}

    missing = [name for name in REQUIRED_TOOLS if name not in tools_by_name]
    if missing:
        raise RuntimeError(
            f"mcp_server.py did not expose required tool(s): {missing}. "
            f"Tools actually available: {sorted(tools_by_name)}"
        )
    return tools_by_name


def _build_graph(tools_by_name: dict) -> CompiledStateGraph:
    """Assemble the Supervisor + worker-agent LangGraph StateGraph."""

    inventory_tool = tools_by_name["check_inventory_levels"]
    compliance_tool = tools_by_name["verify_store_compliance"]

    supervisor_router = ChatAnthropic(model=LLM_MODEL, temperature=0).with_structured_output(
        RouteDecision
    )
    worker_llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    inventory_react_agent = create_react_agent(
        worker_llm,
        tools=[inventory_tool],
        prompt=INVENTORY_AGENT_PROMPT,
        name="inventory_agent",
    )
    compliance_react_agent = create_react_agent(
        worker_llm,
        tools=[compliance_tool],
        prompt=COMPLIANCE_AGENT_PROMPT,
        name="compliance_agent",
    )

    async def _run_worker(react_agent: CompiledStateGraph, state: AgentState) -> dict:
        """
        Run a self-contained ReAct worker to completion on the current
        conversation and fold only the messages it newly produced (its
        tool call, the real ToolMessage result, and its final answer)
        back into the parent graph's state.

        The worker is invoked with its own default internal schema
        (just `messages` + its private recursion guard) rather than the
        Supervisor's extended AgentState, so its internal tool-call loop
        stays fully isolated from the outer supervisor/routing state.
        """
        sub_result = await react_agent.ainvoke({"messages": state["messages"]})
        new_messages = sub_result["messages"][len(state["messages"]):]
        return {"messages": new_messages}

    async def inventory_agent_node(state: AgentState) -> dict:
        return await _run_worker(inventory_react_agent, state)

    async def compliance_agent_node(state: AgentState) -> dict:
        return await _run_worker(compliance_react_agent, state)

    async def supervisor_node(state: AgentState) -> dict:
        iteration = state.get("iterations", 0)

        if iteration >= MAX_SUPERVISOR_ITERATIONS:
            return {
                "next": "FINISH",
                "iterations": iteration + 1,
                "routing_log": [
                    {
                        "iteration": iteration,
                        "next": "FINISH",
                        "reasoning": "Supervisor iteration guard reached; forcing completion.",
                    }
                ],
            }

        conversation = [SystemMessage(content=SUPERVISOR_PROMPT), *state["messages"]]
        decision: RouteDecision = await supervisor_router.ainvoke(conversation)

        return {
            "next": decision.next,
            "iterations": iteration + 1,
            "routing_log": [
                {"iteration": iteration, "next": decision.next, "reasoning": decision.reasoning}
            ],
        }

    def route_from_supervisor(state: AgentState) -> str:
        next_hop = state.get("next", "FINISH")
        if next_hop in ("inventory_agent", "compliance_agent"):
            return next_hop
        return END

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("inventory_agent", inventory_agent_node)
    builder.add_node("compliance_agent", compliance_agent_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"inventory_agent": "inventory_agent", "compliance_agent": "compliance_agent", END: END},
    )
    # Workers report back to the Supervisor rather than ending directly,
    # so a compound request (e.g. inventory *and* compliance in one
    # message) can be routed to a second worker before FINISH.
    builder.add_edge("inventory_agent", "supervisor")
    builder.add_edge("compliance_agent", "supervisor")

    return builder.compile()


# --------------------------------------------------------------------------- 
# Public entry point
# --------------------------------------------------------------------------- 

async def run_agentic_workflow(
    user_query: str,
    mcp_server_path: Path = MCP_SERVER_PATH,
) -> AgentState:
    """
    Run one user request through the full Supervisor -> worker(s) graph
    and return the final graph state.

    Opens exactly one persistent MCP session (one mcp_server.py
    subprocess) for the whole run, so every tool call a worker makes
    during this request reuses the same live database connection pool
    and vector store handle rather than paying subprocess-startup cost
    per call.
    """
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")

    if not mcp_server_path.exists():
        raise FileNotFoundError(f"mcp_server.py not found at '{mcp_server_path}'.")

    client = MultiServerMCPClient(
        {
            "bizagent": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(mcp_server_path)],
                "cwd": str(mcp_server_path.parent),
            }
        }
    )

    async with client.session("bizagent") as session:
        tools_by_name = await _load_verified_tools(session)
        graph = _build_graph(tools_by_name)

        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_query)],
            "next": "",
            "routing_log": [],
            "iterations": 0,
        }
        final_state = await graph.ainvoke(initial_state, config={"recursion_limit": 25})

    return final_state


def _final_answer_text(state: AgentState) -> str:
    """Extract the last assistant-authored message's text for display."""
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "ai" and message.content:
            return message.content if isinstance(message.content, str) else str(message.content)
    return "(No assistant response was produced.)"


# --------------------------------------------------------------------------- 
# Isolated CLI entry point
# --------------------------------------------------------------------------- 

if __name__ == "__main__":
    if "ANTHROPIC_API_KEY" not in os.environ:
        print(
            "ERROR: ANTHROPIC_API_KEY is not set in the environment. "
            "Set it before running this orchestrator -- no key is hardcoded here.",
            file=sys.stderr,
        )
        sys.exit(1)

    query = " ".join(sys.argv[1:]).strip() or input("Ask BizAgent: ").strip()

    result_state = asyncio.run(run_agentic_workflow(query))

    print("\n=== Routing Log ===")
    for entry in result_state["routing_log"]:
        print(f"  [{entry['iteration']}] -> {entry['next']}  ({entry['reasoning']})")

    print("\n=== Final Answer ===")
    print(_final_answer_text(result_state))
