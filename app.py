import streamlit as st
import os
import sys
from pathlib import Path

# --- Ensure API keys reach os.environ no matter what --------------------
# Streamlit Cloud is supposed to auto-expose root-level secrets.toml keys
# as environment variables, but this has proven unreliable in practice.
# This makes it explicit and guaranteed: whatever is in st.secrets gets
# copied into os.environ *before* agents.py (and the LLM clients it
# constructs) ever look for it.
for _key in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    try:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
    except Exception:
        # st.secrets raises if no secrets.toml exists at all (e.g. local
        # dev without one) -- that's fine, just skip syncing in that case.
        pass

from langchain_core.messages import HumanMessage, AIMessage

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from database import DatabaseManager
from agents import graph_pipeline

st.set_page_config(
    page_title="BizAgent AI - Operations Control Center",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FIX: Database instance ko state cache mein handle karne ke liye ---
@st.cache_resource
def get_db_instance():
    db = DatabaseManager()
    db.initialize()
    return db

db_global = get_db_instance()
# ----------------------------------------------------------------------

POLICIES_DIR = BASE_DIR / "policies"
POLICIES_DIR.mkdir(exist_ok=True)

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def fetch_live_dashboard_stats():
    try:
        with db_global.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) as total FROM products;")
            row_total = cur.fetchone()
            total_p = row_total["total"] if row_total else 0
            
            cur.execute("SELECT COUNT(*) as low_count FROM products WHERE stock_level < minimum_required_stock;")
            row_low = cur.fetchone()
            critical_low = row_low["low_count"] if row_low else 0
            
            cur.execute("SELECT COUNT(*) as sales_count FROM sales_transactions;")
            row_sales = cur.fetchone()
            total_sales = row_sales["sales_count"] if row_sales else 0
            
            return total_p, critical_low, total_sales
    except Exception:
        return 0, 0, 0


def render_security_portal():
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🏪 BizAgent AI Supermart Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6B7280;'>Secure Manager Identity Terminal Gateway</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_tab, signup_tab = st.tabs(["🔑 Login Area", "📝 Create Manager Account"])
        
        with signup_tab:
            st.subheader("Register New Store Account")
            new_user = st.text_input("Choose Username", key="reg_user")
            new_pass = st.text_input("Choose Secure Password", type="password", key="reg_pass")
            signup_btn = st.button("Register Identity System")
            
            if signup_btn:
                if new_user and new_pass:
                    # Global single instance use ho raha hai
                    success = db_global.register_user(new_user, new_pass)
                    if success:
                        st.success("✅ Account registered permanently! Now go to the Login Area to access the board.")
                    else:
                        st.error("❌ Username already exists in the system database.")
                else:
                    st.warning("⚠️ Please fill in all credentials fields.")
                    
        with login_tab:
            st.subheader("Authorized Operator Entry")
            login_user = st.text_input("Operator Username", key="log_user")
            login_pass = st.text_input("Password", type="password", key="log_pass")
            login_btn = st.button("Verify Identity Key")
            
            if login_btn:
                # Same shared instance read karega
                is_valid = db_global.verify_user_credentials(login_user, login_pass)
                if is_valid:
                    st.session_state["authenticated"] = True
                    st.success("Access Granted. Initializing console...")
                    st.rerun()
                else:
                    st.error("❌ Access Denied: Invalid credentials matching database records.")


if not st.session_state["authenticated"]:
    render_security_portal()
else:
    st.sidebar.markdown("<h2 style='color: #1E3A8A;'>🛡️ BizAgent Control</h2>", unsafe_allow_html=True)
    st.sidebar.info("Operational Status: Active Connected Gateway.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 Data & Policy Ingestion")
    uploaded_csv = st.sidebar.file_uploader("Upload Store Inventory CSV", type=["csv"])
    if uploaded_csv is not None:
        temp_csv = BASE_DIR / "temp_products.csv"
        with open(temp_csv, "wb") as f:
            f.write(uploaded_csv.getbuffer())
        try:
            db_global.seed_from_csv(temp_csv)
            st.sidebar.success("✅ Inventory Database Synchronized!")
            os.remove(temp_csv)
        except Exception as e:
            st.sidebar.error(f"Ingestion Failure: {str(e)}")

    uploaded_pdf = st.sidebar.file_uploader("Upload Policy PDF Manual", type=["pdf"])
    if uploaded_pdf is not None:
        with open(POLICIES_DIR / uploaded_pdf.name, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.sidebar.success(f"✅ Ingested: {uploaded_pdf.name}")

    total_sku, critical_low, total_sales = fetch_live_dashboard_stats()
    st.sidebar.markdown("---")
    st.sidebar.subheader("📉 Live Supermart Analytics")
    st.sidebar.metric(label="Total Tracked Products (SKUs)", value=total_sku)
    st.sidebar.metric(label="Critical Low Stock Alerts", value=critical_low)
    st.sidebar.metric(label="Historical Sales Items Handled", value=total_sales)
    
    if st.sidebar.button("Logout Gateway"):
        st.session_state["authenticated"] = False
        st.session_state["chat_history"] = []
        st.rerun()

    st.markdown("<h1>📊 Operational Agent Console</h1>", unsafe_allow_html=True)
    st.caption("Central multi-agent supervisor running dynamic runtime SQLite and uploaded PDF analytics workflows.")
    
    for message in st.session_state["chat_history"]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    if user_query := st.chat_input("Ask about stock levels, products list, or compliance policy checks..."):
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state["chat_history"].append(HumanMessage(content=user_query))
        
        with st.chat_message("assistant"):
            with st.spinner("Supervisor Agent routing workflow context..."):
                try:
                    inputs = {"messages": st.session_state["chat_history"]}
                    result = graph_pipeline.invoke(inputs)
                    final_reply = result["messages"][-1].content
                    st.write(final_reply)
                    st.session_state["chat_history"].append(AIMessage(content=final_reply))
                except Exception as error:
                    st.error(f"System Workflow Exception: {str(error)}")







