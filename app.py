"""
app.py
======
BizAgent - Supermart Operations Console (Streamlit UI)

ZERO IN-LINE MOCKING POLICY
----------------------------
- No hardcoded mock lists/dicts/sample transactions live in this file.
- All data comes from `DatabaseManager` (database.py) querying the real
  `supermart_ops.db` SQLite file, via parameterized queries only.
- All credentials and business rules (escalation thresholds, assistant
  intent -> query mappings) are read from external flat files (.env and
  sop_policy.txt) through config.py -- never embedded in this module.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from config import AppConfig, Credential, SOPPolicy, load_app_config, load_sop_policy
from database import DatabaseManager

# --------------------------------------------------------------------------- 
# Page setup
# --------------------------------------------------------------------------- 

st.set_page_config(page_title="BizAgent Operations Console", page_icon="🛒", layout="wide")


# --------------------------------------------------------------------------- 
# Cached, process-wide singletons: config, policy, and the DB connection pool
# --------------------------------------------------------------------------- 

@st.cache_resource
def get_config() -> AppConfig:
    return load_app_config()


@st.cache_resource
def get_policy(_config: AppConfig) -> SOPPolicy:
    return load_sop_policy(_config.sop_policy_file)


@st.cache_resource
def get_db(_config: AppConfig) -> DatabaseManager:
    """
    One pooled DatabaseManager for the lifetime of the Streamlit process.
    `_config` is prefixed with an underscore so Streamlit's cache_resource
    does not try (and fail) to hash it.
    """
    manager = DatabaseManager(db_path=_config.db_path)
    manager.initialize()
    # Only seed if the operator hasn't populated the DB yet -- never
    # re-seed over live operational data.
    if manager.row_counts()["products"] == 0:
        manager.seed_database()
    return manager


def rows_to_df(rows: list) -> pd.DataFrame:
    """Convert a list of sqlite3.Row objects into a DataFrame."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


# --------------------------------------------------------------------------- 
# Authentication (credentials sourced from .env via config.py)
# --------------------------------------------------------------------------- 

def render_login(config: AppConfig) -> None:
    st.title("🛒 " + config.app_title)
    st.caption("Sign in with your operational credentials to continue.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        credential: Optional[Credential] = config.authenticate(username, password)
        if credential:
            st.session_state["auth"] = {"username": credential.username, "role": credential.role}
            st.rerun()
        else:
            st.error("Invalid username or password.")


def require_login(config: AppConfig) -> Optional[dict]:
    if "auth" not in st.session_state:
        render_login(config)
        return None
    return st.session_state["auth"]


# --------------------------------------------------------------------------- 
# Tab: Overview
# --------------------------------------------------------------------------- 

def render_overview(db: DatabaseManager, config: AppConfig, policy: SOPPolicy) -> None:
    st.subheader("Live Operational Snapshot")

    inv = db.get_inventory_summary()
    sales = db.get_sales_summary(days=config.sales_lookback_days)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active SKUs", int(inv["total_skus"]))
    col2.metric("Units on Hand", int(inv["total_units"]))
    col3.metric("Inventory Value", f"${inv['inventory_value']:,.2f}")
    col4.metric("Low-Stock SKUs", int(inv["low_stock_count"]))

    col5, col6, col7 = st.columns(3)
    col5.metric(f"Receipts (last {config.sales_lookback_days}d)", int(sales["receipt_count"]))
    col6.metric(f"Units Sold (last {config.sales_lookback_days}d)", int(sales["units_sold"]))
    col7.metric(f"Revenue (last {config.sales_lookback_days}d)", f"${sales['revenue']:,.2f}")

    st.divider()
    st.subheader(f"Top Sellers - Last {config.top_products_lookback_days} Days")
    top_df = rows_to_df(
        db.get_top_selling_products(
            days=config.top_products_lookback_days, limit=config.top_products_limit
        )
    )
    if top_df.empty:
        st.info("No sales recorded in this window yet.")
    else:
        chart_df = top_df.set_index("name")[["units_sold"]]
        st.bar_chart(chart_df)
        st.dataframe(top_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- 
# Tab: Inventory
# --------------------------------------------------------------------------- 

def render_inventory(db: DatabaseManager) -> None:
    st.subheader("Product Inventory")

    categories = ["All categories"] + db.get_categories()
    col_a, col_b = st.columns([1, 2])
    with col_a:
        category = st.selectbox("Filter by category", categories)
    with col_b:
        search_term = st.text_input("Search by name or SKU", placeholder="e.g. rice, SKU-1001")

    if search_term:
        products = db.search_products(search_term)
    elif category != "All categories":
        products = db.get_all_products(category=category)
    else:
        products = db.get_all_products()

    df = rows_to_df(products)
    if df.empty:
        st.warning("No matching products found.")
    else:
        display_cols = [
            "sku", "barcode", "name", "category", "stock_level",
            "minimum_required_stock", "wholesale_price", "retail_price", "updated_at",
        ]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("⚠️ At or Below Minimum Stock")
    low_df = rows_to_df(db.get_low_stock_products())
    if low_df.empty:
        st.success("No products are currently below their minimum stock threshold.")
    else:
        st.dataframe(low_df, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- 
# Tab: Sales
# --------------------------------------------------------------------------- 

def render_sales(db: DatabaseManager, config: AppConfig) -> None:
    st.subheader("Recent Transactions")
    recent_df = rows_to_df(db.get_recent_transactions(limit=config.recent_transactions_limit))
    if recent_df.empty:
        st.info("No sales transactions recorded yet.")
    else:
        st.dataframe(recent_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Record a New Sale")
    st.caption("This writes directly to sales_transactions and decrements live stock.")
    products = db.get_all_products()
    if not products:
        st.warning("No products available to sell.")
        return

    sku_options = {f"{p['sku']} - {p['name']} (on hand: {p['stock_level']})": p["sku"] for p in products}
    with st.form("record_sale_form"):
        selected_label = st.selectbox("Product", list(sku_options.keys()))
        quantity = st.number_input("Quantity", min_value=1, step=1, value=1)
        submitted = st.form_submit_button("Record Sale")

    if submitted:
        sku = sku_options[selected_label]
        receipt_id = f"RCT-{int(datetime.now().timestamp())}"
        try:
            db.record_sale(receipt_id=receipt_id, sku=sku, quantity=int(quantity))
            st.success(f"Sale recorded under receipt {receipt_id}.")
            st.rerun()
        except (ValueError, sqlite3.Error) as exc:
            st.error(f"Could not record sale: {exc}")


# --------------------------------------------------------------------------- 
# Tab: Vendors & Reorder
# --------------------------------------------------------------------------- 

def render_vendors(db: DatabaseManager, policy: SOPPolicy, role: str) -> None:
    st.subheader("Reorder Recommendations")
    reorder_df = rows_to_df(db.get_reorder_recommendations())
    if reorder_df.empty:
        st.success("No open reorder recommendations right now.")
    else:
        reorder_df["priority"] = reorder_df["lead_time_days"].apply(
            lambda days: "HIGH PRIORITY"
            if pd.notna(days) and days > policy.escalation_lead_time_days
            else "Standard"
        )
        if role != "Operations Manager":
            # SOP access policy: associates see that a reorder is needed,
            # not which vendor / contract terms apply.
            st.dataframe(
                reorder_df[["sku", "name", "category", "stock_level", "minimum_required_stock", "priority"]],
                use_container_width=True, hide_index=True,
            )
            st.caption("Vendor and lead-time detail is restricted to the Operations Manager role.")
        else:
            st.dataframe(reorder_df, use_container_width=True, hide_index=True)

    st.divider()
    if role == "Operations Manager":
        st.subheader("Vendor Contracts")
        vendor_df = rows_to_df(db.get_vendor_contracts())
        st.dataframe(vendor_df, use_container_width=True, hide_index=True)
    else:
        st.info("Sign in as an Operations Manager to view vendor contract terms.")


# --------------------------------------------------------------------------- 
# Tab: Assistant (intent -> live query, driven entirely by sop_policy.txt)
# --------------------------------------------------------------------------- 

def _dispatch_intent(db: DatabaseManager, config: AppConfig, method: str):
    """Map a policy-declared method name to a live, parameterized DB call."""
    dispatch_table = {
        "get_reorder_recommendations": lambda: db.get_reorder_recommendations(),
        "get_inventory_summary": lambda: db.get_inventory_summary(),
        "get_sales_summary": lambda: db.get_sales_summary(days=config.sales_lookback_days),
        "get_top_selling_products": lambda: db.get_top_selling_products(
            days=config.top_products_lookback_days, limit=config.top_products_limit
        ),
        "get_vendor_contracts": lambda: db.get_vendor_contracts(),
        "get_recent_transactions": lambda: db.get_recent_transactions(
            limit=config.recent_transactions_limit
        ),
    }
    if method not in dispatch_table:
        raise KeyError(f"sop_policy.txt references unknown method '{method}'")
    return dispatch_table[method]()


def _summarize_result(intent_name: str, result) -> str:
    """Build a plain-language summary from the *actual* query result."""
    if isinstance(result, sqlite3.Row):
        row = dict(result)
        if intent_name == "inventory":
            return (
                f"You have **{row['total_skus']} SKUs** on hand totaling "
                f"**{row['total_units']} units** (${row['inventory_value']:,.2f} at retail). "
                f"**{row['low_stock_count']}** are at or below their minimum stock level."
            )
        if intent_name == "sales":
            return (
                f"In the reporting window there were **{row['receipt_count']} receipts**, "
                f"**{row['units_sold']} units sold**, for **${row['revenue']:,.2f}** in revenue."
            )
        return str(row)

    rows = list(result) if result else []
    if not rows:
        return "No matching records were found for that query."

    if intent_name == "reorder":
        return f"**{len(rows)} product(s)** are at or below their minimum stock threshold."
    if intent_name == "top_sellers":
        names = ", ".join(dict(r)["name"] for r in rows[:3])
        return f"Top performers right now: **{names}**."
    if intent_name == "vendors":
        return f"There are **{len(rows)} active vendor contract(s)** on file."
    if intent_name == "recent_sales":
        return f"Here are the **{len(rows)} most recent** sale line-items."
    return f"Found **{len(rows)}** matching record(s)."


def render_assistant(db: DatabaseManager, config: AppConfig, policy: SOPPolicy) -> None:
    st.subheader("Operations Assistant")
    st.caption(
        "Ask about stock, sales, vendors, or reorders. Every response below is generated "
        "live from supermart_ops.db -- the assistant only knows what the query returns."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("table") is not None and not turn["table"].empty:
                st.dataframe(turn["table"], use_container_width=True, hide_index=True)

    question = st.chat_input("e.g. 'what needs to be reordered?' or 'top sellers this month'")
    if question:
        intent = policy.match_intent(question)
        if intent is None:
            available = ", ".join(sorted({i.name.replace("_", " ") for i in policy.intents}))
            answer = (
                "I couldn't match that to a known operation. Try asking about: " + available + "."
            )
            table = pd.DataFrame()
        else:
            result = _dispatch_intent(db, config, intent.method)
            answer = _summarize_result(intent.name, result)
            table = (
                rows_to_df([result]) if isinstance(result, sqlite3.Row) else rows_to_df(list(result))
            )
        st.session_state["chat_history"].append({"question": question, "answer": answer, "table": table})
        st.rerun()


# --------------------------------------------------------------------------- 
# SOP policy viewer (sidebar)
# --------------------------------------------------------------------------- 

def render_policy_sidebar(policy: SOPPolicy, auth: dict, config: AppConfig) -> None:
    st.sidebar.title("BizAgent")
    st.sidebar.write(f"Signed in as **{auth['username']}**")
    st.sidebar.write(f"Role: **{auth['role']}**")
    if st.sidebar.button("Sign out"):
        st.session_state.pop("auth", None)
        st.session_state.pop("chat_history", None)
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(f"Escalation threshold: > {policy.escalation_lead_time_days} lead-time days")
    st.sidebar.caption(f"Slow-moving window: {policy.slow_moving_window_days} days")
    with st.sidebar.expander("SOP Policy (sop_policy.txt)"):
        st.write(policy.narrative)


# --------------------------------------------------------------------------- 
# Main
# --------------------------------------------------------------------------- 

def main() -> None:
    config = get_config()
    policy = get_policy(config)
    db = get_db(config)

    auth = require_login(config)
    if auth is None:
        return

    render_policy_sidebar(policy, auth, config)

    st.title("🛒 " + config.app_title)

    tab_overview, tab_inventory, tab_sales, tab_vendors, tab_assistant = st.tabs(
        ["Overview", "Inventory", "Sales", "Vendors & Reorder", "Assistant"]
    )

    with tab_overview:
        render_overview(db, config, policy)
    with tab_inventory:
        render_inventory(db)
    with tab_sales:
        render_sales(db, config)
    with tab_vendors:
        render_vendors(db, policy, auth["role"])
    with tab_assistant:
        render_assistant(db, config, policy)


if __name__ == "__main__":
    main()
