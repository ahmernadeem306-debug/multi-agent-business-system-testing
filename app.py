"""
app.py
======
Enterprise-grade Streamlit application dashboard for BizAgent.
Features runtime file uploaders for CSV inventory data and Policy PDFs,
bypassing any hardcoded local asset dependencies.
"""

import streamlit as st
import os
import shutil
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

# Direct imports from your existing modular files
from database import DatabaseManager
from rag_engine import query_knowledge_base
from agents import graph_pipeline

# Configure Page Branding layout metrics
st.set_page_config(
    page_title="BizAgent AI - Operations Control Center",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------- 
# Configuration & Local Directories Setup
# --------------------------------------------------------------------------- 
BASE_DIR = Path(__file__).resolve().parent
POLICIES_DIR = BASE_DIR / "policies"
POLICIES_DIR.mkdir(exist_ok=True)  # Create runtime folder if not present

# --------------------------------------------------------------------------- 
# Real-Time Dynamic Metrics Processing
# --------------------------------------------------------------------------- 
def fetch_live_dashboard_stats():
    """Queries your actual database file to track active inventory anomalies."""
    db = DatabaseManager()
    db.initialize()
    
    with db.get_cursor() as cur:
        # Fetching Total Number of Registered Products
        cur.execute("SELECT COUNT(*) as total FROM products;")
        total_p = cur.fetchone()["total"]
        
        # Fetching Count of Low-Stock Operational Triggers
        cur.execute("SELECT COUNT(*) as low_count FROM products WHERE stock_level < minimum_required_stock;")
        low_p = cur.fetchone()["low_count"]
        
        # Fetching Total Sales Volume Recorded in System
        cur.execute("SELECT IFNULL(SUM(quantity), 0) as total_sales FROM sales_transactions;")
        sales_v = cur.fetchone()["total_sales"]
        
    return total_p, low_p, sales_v

# --------------------------------------------------------------------------- 
# Secure Local Authentication Logic Barrier
# --------------------------------------------------------------------------- 
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def render_login_portal():
    """Builds an isolated login screen wrapper."""
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏪 BizAgent AI Supermart Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280;'>Enterprise Dashboard Gateway Security Terminal</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns()
    with col2:
        with st.form("security_handshake"):
            st.subheader("Administrative Log-In")
            username = st.text_input("Operator Username")
            password = st.text_input("Secure Access Key", type="password")
            submit = st.form_submit_button("Verify Identity")
            
            if submit:
                if username == "admin" and password == "supermart2026":
                    st.session_state["authenticated"] = True
                    st.success("Identity Verified successfully. Initializing system panels...")
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid operator credentials.")

# --------------------------------------------------------------------------- 
# Main Enterprise Dashboard Rendering
# --------------------------------------------------------------------------- 
if not st.session_state["authenticated"]:
    render_login_portal()
else:
    # --- Sidebar Live Diagnostics & Data Ingestion System Panel ---
    st.sidebar.markdown("<h2 style='color: #1E3A8A;'>🛡️ BizAgent Control</h2>", unsafe_allow_html=True)
    st.sidebar.info("Operational Status: Active Gateway Connected.")
    
    # --- RUNTIME DATA UPLOAD SECTION ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Data & Policy Ingestion")
    
    # 1. Product Inventory CSV Uploader
    uploaded_csv = st.sidebar.file_uploader("Upload Store Inventory CSV", type=["csv"])
    if uploaded_csv is not None:
        temp_csv_path = BASE_DIR / "temp_uploaded_products.csv"
        # Save file locally for database seeding processing
        with open(temp_csv_path, "wb") as f:
            f.write(uploaded_csv.getbuffer())
        
        try:
            db = DatabaseManager()
            db.initialize()
            db.seed_from_csv(temp_csv_path)
            st.sidebar.success("✅ Products Database Seeded/Updated!")
            os.remove(temp_csv_path)  # Clean up temp file
        except Exception as e:
            st.sidebar.error(f"Database Ingestion Error: {str(e)}")

    # 2. Company Compliance PDF Uploader
    uploaded_pdf = st.sidebar.file_uploader("Upload Policy PDF Manual", type=["pdf"])
    if uploaded_pdf is not None:
        save_pdf_path = POLICIES_DIR / uploaded_pdf.name
        with open(save_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
            
        st.sidebar.success(f"✅ Saved: {uploaded_pdf.name}")
        st.sidebar.info("Run indexing via background console tool if required.")
        # Note: Your background agent triggers rag_engine logic directly over the policies directory.

    # Executing dynamic calculations directly from database logic
    total_sku, critical_low, total_sales = fetch_live_dashboard_stats()
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📉 Live Supermart Analytics")
    st.sidebar.metric(label="Total Tracked Products (SKUs)", value=total_sku)
    st.sidebar.metric(label="Critical Low Stock Alerts", value=critical_low, delta=f"{critical_low} items short" if critical_low > 0 else "0 anomalies")
    st.sidebar.metric(label="Historical Sales Items Handled", value=total_sales)
    
    # Safe Sign-Out Command Protocol
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout System Gateway"):
        st.session_state["authenticated"] = False
        st.session_state["chat_history"] = []
        st.rerun()

    # --- Main System Command Pane Panel ---
    st.markdown("<h1>📊 Operational Agent Console</h1>", unsafe_allow_html=True)
    st.caption("Central multi-agent supervisor running dynamic runtime SQLite and uploaded PDF analytics workflows.")
    
    # Display Persistent Chat History State Blocks
    for message in st.session_state["chat_history"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    # Monitor user runtime inputs
    if user_query := st.chat_input("Ask about stock levels, products list, or compliance policy checks..."):
        
        # Instantly append user input interface container block
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state["chat_history"].append(HumanMessage(content=user_query))
        
        # Process context streaming directly inside the multi-agent graph layout
        with st.chat_message("assistant"):
            with st.spinner("Supervisor Agent routing workflow context to sub-modules..."):
                try:
                    # Packaging conversational block context matching state parameters
                    inputs = {"messages": st.session_state["chat_history"]}
                    
                    # Run the active production-grade LangGraph compiled pipeline file
                    result = graph_pipeline.invoke(inputs)
                    
                    # Extract the absolute final synthetic response string layer
                    final_agent_reply = result["messages"][-1].content
                    
                    # Display the execution result bubble instantly
                    st.write(final_agent_reply)
                    st.session_state["chat_history"].append(AIMessage(content=final_agent_reply))
                    
                except Exception as error:
                    st.error(f"System Workflow Exception Interception: {str(error)}")

