"""
mcp_server.py
=============
Model Context Protocol (MCP) server for the BizAgent platform.

Exposes two functional, executable tools to any MCP-compatible client
(Claude Desktop, an agent framework, etc.):

    check_inventory_levels()      -> live low-stock report from database.py
    verify_store_compliance(query) -> live semantic search over the ingested
                                       PDF policies via rag_engine.py

ZERO FAKE DATA
--------------
This file defines no simulated arrays, mock strings, or sample records.
Every tool call executes a real, parameterized query against
`supermart_ops.db` (through the pooled `DatabaseManager`) or the
persisted `./chroma_db` vector store (through `query_knowledge_base`).
If either backing store is empty or unreachable, the tool returns an
empty result / raises a clear error -- it never fabricates a fallback.

Run with:
    python mcp_server.py
        -> starts the server over stdio, ready for an MCP client to attach.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from mcp.server.mcpserver import Context, MCPServer

from database import DatabaseManager
from rag_engine import query_knowledge_base

# --------------------------------------------------------------------------- 
# Configuration
# --------------------------------------------------------------------------- 

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "supermart_ops.db"

DEFAULT_COMPLIANCE_RESULTS = 4  # how many policy chunks to return per query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("bizagent.mcp_server")


# --------------------------------------------------------------------------- 
# Lifespan: one pooled DatabaseManager for the life of the server process
# --------------------------------------------------------------------------- 

@dataclass
class AppContext:
    db: DatabaseManager


@asynccontextmanager
async def app_lifespan(server: MCPServer) -> AsyncIterator[AppContext]:
    """
    Stand up the shared DatabaseManager once when the MCP server starts,
    and release its pooled connections cleanly on shutdown. Tools access
    it via `ctx.request_context.lifespan_context.db` -- never by opening
    a fresh, ad-hoc connection of their own.
    """
    logger.info("Starting BizAgent MCP server -- initializing database connection pool.")
    db = DatabaseManager(db_path=DB_PATH)
    db.initialize()
    try:
        yield AppContext(db=db)
    finally:
        logger.info("Shutting down BizAgent MCP server -- releasing pooled connections.")
        db.shutdown()


# --------------------------------------------------------------------------- 
# Server instance
# --------------------------------------------------------------------------- 

mcp = MCPServer(
    name="bizagent-operations",
    instructions=(
        "Tools for the BizAgent supermart operations platform. "
        "check_inventory_levels reports live low-stock SKUs from the "
        "operational database. verify_store_compliance runs a semantic "
        "search over the store's ingested policy PDFs and returns the "
        "exact matching passages."
    ),
    lifespan=app_lifespan,
)


# --------------------------------------------------------------------------- 
# Tool: check_inventory_levels
# --------------------------------------------------------------------------- 

@mcp.tool()
def check_inventory_levels(ctx: Context) -> list[dict]:
    """
    Return every product currently at or below its minimum required
    stock level, pulled live from the operational database.

    Each result includes: sku, name, category, stock_level,
    minimum_required_stock. Returns an empty list if nothing is
    currently below threshold -- this is a real, possibly-empty query
    result, not a placeholder.
    """
    db: DatabaseManager = ctx.request_context.lifespan_context.db
    rows = db.get_low_stock_products()
    results = [dict(row) for row in rows]
    logger.info("check_inventory_levels -> %d low-stock item(s).", len(results))
    return results


# --------------------------------------------------------------------------- 
# Tool: verify_store_compliance
# --------------------------------------------------------------------------- 

@mcp.tool()
def verify_store_compliance(query: str, ctx: Context | None = None) -> list[dict]:
    """
    Run a semantic search over the store's ingested policy PDFs
    (./policies/*.pdf, indexed into ./chroma_db by rag_engine.py) and
    return the exact matching passages.

    Parameters
    ----------
    query : str
        The compliance question or policy lookup term, e.g.
        "what is the escalation policy for a delayed reorder?"

    Returns
    -------
    list[dict]
        Each dict has keys: content, source_file, page, score. Empty
        list if no PDF passage is a strong enough match, or if the
        knowledge base has not been built yet.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")

    matches = query_knowledge_base(query.strip(), k=DEFAULT_COMPLIANCE_RESULTS)
    logger.info("verify_store_compliance(%r) -> %d matching passage(s).", query, len(matches))
    return matches


# --------------------------------------------------------------------------- 
# Entry point
# --------------------------------------------------------------------------- 

if __name__ == "__main__":
    mcp.run(transport="stdio")
