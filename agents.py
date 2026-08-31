"""
agents.py
=========
Stateful, multi-agent orchestration layer for the BizAgent platform,
built on LangGraph.
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Model used by every agent below. Overridable via env var so you can
# swap models without touching code (e.g. BIZAGENT_LLM_MODEL=gemini-2.0-flash).
# gemini-2.5-flash is on Google AI Studio's free tier (no credit card needed).
import os
LLM_MODEL = os.environ.get("BIZAGENT_LLM_MODEL", "gemini-2.5-flash")


# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str

# Router class for structured output
class RouterDecision(BaseModel):
    next_agent: str = Field(
        description="The next agent to route to. Choose 'inventory', 'compliance', or 'FINISH'."
    )

# 2. Define Agents Nodes
def supervisor_agent(state: AgentState):
    """LLM sets routing logic between teams"""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(RouterDecision)
    
    # System Instruction for routing
    system_prompt = (
        "You are the Supervisor Agent managing a Supermart Control System.\n"
        "Your job is to read the conversation and decide which agent should act next.\n"
        "Options:\n"
        "- If user asks about stock levels, product lists, or inventory data: choose 'inventory'\n"
        "- If user asks about policy, manuals, compliance, or regulations: choose 'compliance'\n"
        "- If the query has been successfully resolved or answered: choose 'FINISH'"
    )
    
    messages = [{"role": "system", "content": system_prompt}] + list(state["messages"])
    decision = structured_llm.invoke(messages)
    
    return {"next_agent": decision.next_agent}

def inventory_agent(state: AgentState):
    """Processes inventory and stock related logic"""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    # Aapka actual inventory tool logic/call yahan automatic run ho sakta hai
    response = llm.invoke([{"role": "system", "content": "You are the Inventory Expert agent. Answer questions about stock levels accurately using Database context."}] + list(state["messages"]))
    return {"messages": [response], "next_agent": "supervisor"}

def compliance_agent(state: AgentState):
    """Processes policy manuals and PDF analytics logic"""
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    # Aapka actual PDF/RAG compliance tool logic yahan execute hoga
    response = llm.invoke([{"role": "system", "content": "You are the Compliance Expert agent. Answer questions about supermart rules and PDF policy manuals."}] + list(state["messages"]))
    return {"messages": [response], "next_agent": "supervisor"}

# 3. Router Edge Logic
def route_next(state: AgentState):
    if state["next_agent"] == "inventory":
        return "inventory"
    elif state["next_agent"] == "compliance":
        return "compliance"
    return END

# 4. Build the LangGraph Workflow Pipeline
workflow = StateGraph(AgentState)

# Add Node Blocks
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("inventory", inventory_agent)
workflow.add_node("compliance", compliance_agent)

# Setup graph edges flow
workflow.add_edge(START, "supervisor")

# Conditional Routing from Supervisor decision
workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {
        "inventory": "inventory",
        "compliance": "compliance",
        END: END
    }
)

# Return loop back to supervisor
workflow.add_edge("inventory", "supervisor")
workflow.add_edge("compliance", "supervisor")

# Compile into executable pipeline graph
graph_pipeline = workflow.compile()

