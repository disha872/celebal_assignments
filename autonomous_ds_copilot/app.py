import os
import streamlit as st
import pandas as pd
import json
import shutil
from dotenv import load_dotenv
from agent import AutonomousDSAgent

def load_and_sanitize_dotenv():
    if os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ[key] = val
        except Exception:
            pass

# Sanitize and load from local .env, overriding system env (handles UTF-8 BOM from PowerShell)
load_and_sanitize_dotenv()
load_dotenv(override=True)

# Page config
st.set_page_config(
    page_title="Autonomous Data Science Co-Pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 50%, #6BCB77 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar CSS */
    .stSidebar {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    
    /* Metric Card Custom Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-val {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 800;
    }
    
    /* Code styling */
    code {
        color: #F43F5E !important;
        font-weight: 600;
    }
    
    /* Status styling */
    .status-container {
        padding: 15px;
        background: #1E293B;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Subtitle
st.title("🤖 Autonomous Data Science Co-Pilot")
st.caption("Agentic AI • Subprocess Sandbox Sandbox • TF-IDF RAG Self-Healing")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/nolan/96/brain.png", width=70)
st.sidebar.header("🔧 Settings & APIs")

llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Gemini", "OpenAI", "Anthropic"],
    index=0
)

# Set model recommendations based on provider
if llm_provider == "Gemini":
    model_name = st.sidebar.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"], index=0)
    api_key_env_val = os.getenv("GEMINI_API_KEY", "")
elif llm_provider == "OpenAI":
    model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    api_key_env_val = os.getenv("OPENAI_API_KEY", "")
else:
    model_name = st.sidebar.selectbox("Model", ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"], index=0)
    api_key_env_val = os.getenv("ANTHROPIC_API_KEY", "")

# Key input
api_key = st.sidebar.text_input(
    f"{llm_provider} API Key",
    type="password",
    value="",
    placeholder="Loaded from .env (Secure)" if api_key_env_val else "Enter API Key...",
    help=f"Enter your {llm_provider} API key. If left blank, we will try to load from environmental variables."
)

max_retries = st.sidebar.slider(
    "Max Self-Healing Retries",
    min_value=1,
    max_value=5,
    value=3,
    help="How many times the agent should fetch documentation and attempt to fix its code if it crashes."
)

# Doc Directory Configuration
docs_directory = "./docs"
agent = None
rag_status_container = st.sidebar.empty()

if docs_directory:
    # Initialize Agent if api_key is available
    resolved_key = (api_key or api_key_env_val).strip()
    if resolved_key:
        try:
            agent = AutonomousDSAgent(
                provider=llm_provider,
                api_key=resolved_key,
                model_name=model_name,
                docs_dir=docs_directory
            )
            # Count indexed docs
            total_chunks = len(agent.rag.chunks)
            rag_status_container.success(f"📚 RAG Index: {total_chunks} chunks loaded\n\n🔑 API Key: Loaded & Active")
        except Exception as e:
            rag_status_container.error(f"Failed to load RAG index: {str(e)}")
    else:
        rag_status_container.warning("⚠️ Enter API Key to activate Agent")

# Sidebar Help & Info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🛠️ Use Case Quick Loads
Click a button below to load a sample dataset and pre-configured query:
""")

# Handle Quick Load Buttons
sample_clicks = {
    "sales": st.sidebar.button("📊 Sales Dashboard", use_container_width=True),
    "audit": st.sidebar.button("🧹 Data Quality Audit", use_container_width=True),
    "trend": st.sidebar.button("📈 Trend Analysis", use_container_width=True),
    "cohort": st.sidebar.button("👥 Cohort Segmentation", use_container_width=True),
    "budget": st.sidebar.button("💼 Operational Budget", use_container_width=True)
}

selected_sample = None
for key, clicked in sample_clicks.items():
    if clicked:
        selected_sample = key

# Main Panel layout
col_left, col_right = st.columns([1, 2], gap="large")

# Use Case Data Loader
# Initialize session state for query and file path
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""
if "uploaded_file_path" not in st.session_state:
    st.session_state["uploaded_file_path"] = None
if "active_filename" not in st.session_state:
    st.session_state["active_filename"] = None

# Sample definitions
sample_files = {
    "sales": "./sample_data/sales_data.csv",
    "audit": "./sample_data/dirty_data.csv",
    "trend": "./sample_data/traffic_data.json",
    "cohort": "./sample_data/customer_segments.csv",
    "budget": "./sample_data/ad_hoc_queries.xlsx"
}

sample_queries = {
    "sales": "Create a bar chart showing the total revenue generated by each region.",
    "audit": "Identify duplicate rows, clean the bad salary values, handle missing ages by filling with the mean, and write a summary of changes.",
    "trend": "Is my traffic growing? Create a line chart showing daily Visitors and PageViews over time.",
    "cohort": "Segment customers by Spending Score vs Annual Income. Generate a scatter plot showing these groups.",
    "budget": "Compare the budgets and actual spent for each department from the Budgets and Expenses sheets. Draw a grouped bar chart."
}

# Check if sample generated
if not os.path.exists("./sample_data"):
    # If sample folders are not generated, generate them silently
    try:
        from generate_sample_data import make_sample_data
        make_sample_data()
    except Exception:
        pass

if selected_sample:
    st.session_state["active_filename"] = os.path.basename(sample_files[selected_sample])
    st.session_state["uploaded_file_path"] = sample_files[selected_sample]
    st.session_state["user_query"] = sample_queries[selected_sample]

# Read active variables from state
uploaded_file_path = st.session_state["uploaded_file_path"]
active_filename = st.session_state["active_filename"]

if selected_sample:
    st.info(f"Loaded Sample Use Case: **{active_filename}**")

with col_left:
    st.subheader("📂 Upload Data File")
    user_file = st.file_uploader(
        "Upload CSV, Excel, or JSON",
        type=["csv", "xlsx", "xls", "json"],
        help="Upload the file you want the agent to analyze."
    )
    
    if user_file is not None:
        # Save uploaded file to temp file
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        uploaded_file_path = os.path.join(temp_dir, user_file.name)
        with open(uploaded_file_path, "wb") as f:
            f.write(user_file.getbuffer())
        active_filename = user_file.name
        st.session_state["uploaded_file_path"] = uploaded_file_path
        st.session_state["active_filename"] = active_filename
    else:
        # If user cleared the file uploader manually, reset the session state ONLY IF it was a manually uploaded file
        if st.session_state["uploaded_file_path"] and "temp_uploads" in str(st.session_state["uploaded_file_path"]):
            st.session_state["uploaded_file_path"] = None
            st.session_state["active_filename"] = None
            uploaded_file_path = None
            active_filename = None
        
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        # File inspection
        ext = os.path.splitext(uploaded_file_path)[1].lower()
        file_size = os.path.getsize(uploaded_file_path) / 1024 # KB
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Active File</div>
            <div class="metric-val">{active_filename}</div>
            <span style='color: #4D96FF;'>Size: {file_size:.2f} KB</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Load sample to show columns
        try:
            if ext == '.csv':
                df_preview = pd.read_csv(uploaded_file_path, nrows=5)
            elif ext in ['.xlsx', '.xls']:
                xl = pd.ExcelFile(uploaded_file_path)
                df_preview = pd.read_excel(uploaded_file_path, sheet_name=xl.sheet_names[0], nrows=5)
                st.write(f"Sheets available: `{xl.sheet_names}`")
            elif ext == '.json':
                with open(uploaded_file_path, 'r', encoding='utf-8') as f:
                    js_data = json.load(f)
                df_preview = pd.DataFrame(js_data[:5] if isinstance(js_data, list) else js_data)
            
            with st.expander("🔍 Preview Columns & Data Type"):
                st.dataframe(df_preview)
                st.write("Column details:")
                st.code(df_preview.dtypes.to_string())
        except Exception as e:
            st.error(f"Error previewing file: {str(e)}")

with col_right:
    st.subheader("💬 Ask Your Question")
    user_query = st.text_area(
        "Enter your query in plain English:",
        value=st.session_state["user_query"],
        height=100,
        placeholder="e.g., 'Find outliers in the salary column and plot them' or 'Show a trend of traffic daily'"
    )
    st.session_state["user_query"] = user_query
    
    submit_btn = st.button("🚀 Run Agentic Analysis", type="primary", use_container_width=True)
    
    if submit_btn:
        if not uploaded_file_path:
            st.warning("⚠️ Please upload a data file or select a use case from the sidebar first.")
        elif not user_query.strip():
            st.warning("⚠️ Please enter a question or query describing the analysis.")
        elif agent is None:
            st.error("❌ Agent not initialized. Please make sure you have provided an API key in the sidebar.")
        else:
            # Let's run the Agent Loop!
            output_dir = "./sandbox_runs"
            os.makedirs(output_dir, exist_ok=True)
            
            status_container = st.status("🧠 Starting Autonomous Data Science Agent...", expanded=True)
            
            # Custom callback status update
            # We can capture progress steps by executing the agent and displaying them
            try:
                with status_container:
                    st.write("📁 Extracting schema and sample rows...")
                    # Run the self-healing agent loop
                    result = agent.run(
                        file_path=uploaded_file_path,
                        user_query=user_query,
                        output_dir=output_dir,
                        max_retries=max_retries
                    )
                    
                    # Log the steps inside the status container
                    for step in result.get("steps", []):
                        attempt = step["attempt"]
                        st.write(f"---")
                        st.write(f"🤖 **Attempt {attempt}**")
                        
                        if "code" in step:
                            with st.expander(f"📝 Generated Python Code (Attempt {attempt})"):
                                st.code(step["code"], language="python")
                                
                        st.write(f"⏳ Executing code in background sandbox...")
                        
                        if step["status"] == "Success":
                            st.write("✅ Execution Succeeded!")
                        elif step["status"] == "Execution Error":
                            st.write(f"❌ Execution crashed with code: `{step['exit_code']}`")
                            with st.expander(f"⚠️ Stderr Output (Attempt {attempt})"):
                                st.code(step["stderr"])
                            
                            # Log RAG details
                            if "retrieved_docs" in step and step["retrieved_docs"]:
                                st.write("📚 **Querying RAG database for fixes:**")
                                for doc in step["retrieved_docs"]:
                                    st.write(f"- Retrieved: `{doc['title']}` from `{doc['file']}`")
                                    
                if result["success"]:
                    status_container.update(label="🎉 Analysis Completed Successfully!", state="complete")
                    
                    # Display output
                    st.success("✅ Analysis successful! Here are the results:")
                    
                    # Display plot if generated
                    plot_path = result.get("plot_path")
                    if plot_path and os.path.exists(plot_path):
                        st.subheader("📊 Visualisation")
                        if plot_path.endswith('.html'):
                            with open(plot_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=500, scrolling=True)
                        else:
                            st.image(plot_path, caption="Generated Visualisation", use_column_width=True)
                            
                    # Display Insights
                    st.subheader("💡 Insights & Summary")
                    st.markdown(result["markdown_insight"])
                    
                    # Clean up temporary uploads if user uploaded
                    if user_file is not None and os.path.exists(uploaded_file_path):
                        os.remove(uploaded_file_path)
                else:
                    status_container.update(label="❌ Analysis Failed.", state="error")
                    st.error(f"Agent failed to deliver a working script: {result['error']}")
                    
            except Exception as ex:
                status_container.update(label="💥 Agent loop error.", state="error")
                st.error(f"An unexpected error occurred during execution: {str(ex)}")
