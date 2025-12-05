import streamlit as st
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from dotenv import load_dotenv

# Import LangChain and Gemini modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Processly | Spec Analytics",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Environment
load_dotenv()
BASE_DIR = os.getcwd()

# --- CUSTOM CSS FOR POLISH ---
st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        h1, h2, h3 {
            color: #0e1117;
        }
        </style>
        """, unsafe_allow_html=True)

# --- GLOBAL VARIABLES ---
api_key = os.getenv("GEMINI_API_KEY")
prompt = None

# Robust Prompt Loading
try:
    with open(os.path.join(BASE_DIR, 'prompt.txt'), "r") as f:
        prompt = f.read()
except FileNotFoundError:
    # Fallback if file not found to prevent crash
    prompt = "Extract the following technical specifications from the battery datasheet."

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.title("💠 Processly")
    st.caption("Intelligent Datasheet Extraction")
    st.markdown("---")

    if not api_key:
        api_key = st.text_input("Google API Key", type="password")
        if not api_key:
            st.warning("⚠️ API Key required to proceed.")
    else:
        st.success("API Key Loaded")

    st.markdown("### Settings")
    st.caption("Model: **gemini-2.5-flash**")

    st.markdown("---")
    # Add a reset button to clear chat history
    if st.button("Clear Session & History", type="primary"):
        st.session_state.messages = []
        if "result" in st.session_state:
            del st.session_state.result
        st.rerun()


# --- 1. DATA SCHEMA (UNCHANGED LOGIC) ---
class BatteryMetric(BaseModel):
    value: Optional[str] = Field(None, description="The final numeric value. Null if not found.")
    unit: Optional[str] = Field(None, description="The unit (e.g., V, A, degC).")
    confidence: float = Field(description="0.0 to 1.0 confidence score.")
    reference: str = Field(description="The exact text snippet used as source.")
    is_calculated: bool = Field(description="True if the AI had to calculate this.")
    calculation_logic: Optional[str] = Field(None, description="Explain the math if calculated.")
    calculation_reason: Optional[str] = Field(None, description="Explain the source of the formula.")


class VoltavisionSpecs(BaseModel):
    # Capacity & Energy
    C_NOMINAL_AH_DB: BatteryMetric = Field(description="Nominal Capacity in Amp-hours")
    E_NOMINAL_WH_DB: BatteryMetric = Field(description="Nominal Energy in Watt-hours")
    # Voltages
    U_MAX_DYN_DB: BatteryMetric = Field(description="Maximum Dynamic Voltage limit")
    U_MIN_DYN_DB: BatteryMetric = Field(description="Minimum Dynamic Voltage limit")
    U_MAX_PULSE_DB: BatteryMetric = Field(description="Maximum Pulse Voltage")
    U_MIN_PULSE_DB: BatteryMetric = Field(description="Minimum Pulse Voltage")
    U_MAX_SAFETY_DB: BatteryMetric = Field(description="Maximum Safety Limit Voltage")
    U_MIN_SAFETY_DB: BatteryMetric = Field(description="Minimum Safety Limit Voltage")
    # Temperatures
    T_MAX_DB: BatteryMetric = Field(description="Maximum operating temperature")
    T_MIN_DB: BatteryMetric = Field(description="Minimum operating temperature")
    T_MAX_TERMINAL_DB: BatteryMetric = Field(description="Max temperature at terminal")
    T_MIN_TERMINAL_DB: BatteryMetric = Field(description="Min temperature at terminal")
    T_MAX_SAFETY_DB: BatteryMetric = Field(description="Max safety temperature")
    T_MIN_SAFETY_DB: BatteryMetric = Field(description="Min safety temperature")
    T_MAX_TERMINAL_SAFETY_DB: BatteryMetric = Field(description="Max safety temp at terminal")
    T_MIN_TERMINAL_SAFETY_DB: BatteryMetric = Field(description="Min safety temp at terminal")
    # Currents
    I_MAX_CHA_DB: BatteryMetric = Field(description="Max Continuous Charge Current")
    I_MAX_DCH_DB: BatteryMetric = Field(description="Max Continuous Discharge Current")
    I_MAX_CHA_SAFETY_DB: BatteryMetric = Field(description="Max Safety Charge Current")
    I_MAX_DCH_SAFETY_DB: BatteryMetric = Field(description="Max Safety Discharge Current")
    I_MAX_CHA_PULSE_DB: BatteryMetric = Field(description="Max Pulse Charge Current")
    I_MAX_DCH_PULSE_DB: BatteryMetric = Field(description="Max Pulse Discharge Current")


# --- 2. EXTRACTION ENGINE (UNCHANGED LOGIC) ---
def analyze_battery_spec(pdf_path, key):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=key)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    # Sending first 15 pages
    full_text = "\n".join([p.page_content for p in pages[:15]])

    structured_llm = llm.with_structured_output(VoltavisionSpecs)

    if not prompt:
        return None, 'Prompt not initialized!'

    system_prompt = prompt + f"""
        ### INPUT TEXT:
        {full_text}
        """
    # Return BOTH the structured data AND the raw text (for the chatbot)
    return structured_llm.invoke(system_prompt), full_text


# --- 3. HELPER FUNCTIONS FOR VISUALIZATION ---
def clean_numeric(val_str):
    """Simple helper to convert string metrics to floats for plotting"""
    try:
        if not val_str: return None
        # Remove common non-numeric chars except . and -
        clean = ''.join(c for c in str(val_str) if c.isdigit() or c in ['.', '-'])
        return float(clean)
    except:
        return None


def create_gauge_chart(value, title, unit, color="blue"):
    fig = go.Figure(go.Indicator(
        mode="number",
        value=value,
        title={"text": title},
        number={'suffix': f" {unit}"},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=30, b=10))
    return fig


# --- 4. MAIN APP LOGIC ---
st.title("💠 Processly Analytics")
st.markdown(
    "Upload your technical documentation to extract specifications, visualize operating windows, and query the document.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_file = st.file_uploader("Upload Datasheet (PDF)", type="pdf")

if uploaded_file and api_key:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # If new file, run extraction
    if "last_file_id" not in st.session_state or st.session_state.last_file_id != file_id:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        status_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("Processing document..."):
            try:
                status_text.text("📖 Reading PDF...")
                progress_bar.progress(25)

                status_text.text("🧠 Analyzing logic...")
                progress_bar.progress(50)

                # Get result AND text
                result, text = analyze_battery_spec(tmp_path, api_key)
                progress_bar.progress(90)

                if not result:
                    st.error(f"Analysis failed: {text}")

                st.session_state.result = result
                st.session_state.pdf_text = text
                st.session_state.last_file_id = file_id

                status_text.text("✅ Complete!")
                progress_bar.progress(100)
                progress_bar.empty()
                status_text.empty()

            except Exception as e:
                st.error(f"Analysis Failed: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    # --- MAIN INTERFACE ---
    if "result" in st.session_state:
        res = st.session_state.result

        # Define Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Data", "📉 Operating Windows", "💬 AI Assistant"])

        # --- PREPARE DATAFRAME ---
        rows = []
        for field_name, metric in res.dict().items():
            if metric:
                rows.append({
                    "Parameter": field_name,
                    "Value": metric.get('value'),
                    "Unit": metric.get('unit', '-'),
                    "Confidence": metric.get('confidence', 0),
                    "Type": "Calculated" if metric.get('is_calculated') else "Extracted",
                    "Source/Logic": metric.get('calculation_logic') if metric.get('is_calculated') else metric.get(
                        'reference', '')
                })
        df = pd.DataFrame(rows)

        # --- TAB 1: DASHBOARD & DATA ---
        with tab1:

            st.subheader("Detailed Specifications")

            # FILTER & SEARCH
            col_search, col_filter = st.columns([3, 1])
            with col_search:
                search_query = st.text_input("🔍 Search Parameter Name", placeholder="e.g. Temperature, Voltage...")
            with col_filter:
                filter_type = st.selectbox("Filter by Type", ["All", "Extracted", "Calculated"])

            # Apply filters
            filtered_df = df.copy()
            if search_query:
                filtered_df = filtered_df[filtered_df['Parameter'].str.contains(search_query, case=False)]
            if filter_type != "All":
                filtered_df = filtered_df[filtered_df['Type'] == filter_type]


            # Style and Show
            def color_confidence(val):
                color = '#d4edda' if val > 0.8 else '#fff3cd' if val > 0.5 else '#f8d7da'
                return f'background-color: {color}'


            st.dataframe(
                filtered_df.style.format({"Confidence": "{:.0%}"}),
                use_container_width=True,
                height=500,
                column_config={
                    "Parameter": st.column_config.TextColumn("Variable", width="medium"),
                    "Source/Logic": st.column_config.TextColumn("Source / Logic", width="large"),
                    "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1,
                                                                  format="%.0f%%"),
                },
                hide_index=True
            )

        # --- TAB 2: GRAPHS ---
        with tab2:
            st.subheader("Operating Windows Visualization")

            g1, g2 = st.columns(2)

            # 1. Voltage Chart
            with g1:
                v_data = {
                    "Category": ["Min Pulse", "Min Dyn", "Max Dyn", "Max Pulse"],
                    "Value": [
                        clean_numeric(res.U_MIN_PULSE_DB.value),
                        clean_numeric(res.U_MIN_DYN_DB.value),
                        clean_numeric(res.U_MAX_DYN_DB.value),
                        clean_numeric(res.U_MAX_PULSE_DB.value)
                    ]
                }
                v_df = pd.DataFrame(v_data).dropna()
                if not v_df.empty:
                    fig_v = px.bar(v_df, x="Category", y="Value", title="Voltage Limits (V)", text_auto=True,
                                   color="Category")
                    fig_v.update_layout(showlegend=False)
                    st.plotly_chart(fig_v, use_container_width=True)
                else:
                    st.info("Insufficient Voltage data for plotting.")

            # 2. Temperature Chart
            with g2:
                t_data = {
                    "Category": ["Min Op", "Max Op", "Min Safety", "Max Safety"],
                    "Value": [
                        clean_numeric(res.T_MIN_DB.value),
                        clean_numeric(res.T_MAX_DB.value),
                        clean_numeric(res.T_MIN_SAFETY_DB.value),
                        clean_numeric(res.T_MAX_SAFETY_DB.value)
                    ]
                }
                t_df = pd.DataFrame(t_data).dropna()
                if not t_df.empty:
                    fig_t = px.bar(t_df, x="Category", y="Value", title="Temperature Limits (°C)", text_auto=True,
                                   color="Value", color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig_t, use_container_width=True)
                else:
                    st.info("Insufficient Temperature data for plotting.")

        # --- TAB 3: CHATBOT ---
        with tab3:
            st.subheader("💬 Chat with Document")

            # Chat Container
            chat_container = st.container(height=500)

            # Display history
            for msg in st.session_state.messages:
                with chat_container.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Input
            if user_input := st.chat_input(
                    "Ask about specific conditions, e.g., 'What is the max safety temperature?'"):
                # Add User Message
                st.session_state.messages.append({"role": "user", "content": user_input})
                with chat_container.chat_message("user"):
                    st.markdown(user_input)

                # Prepare Context
                raw_text = st.session_state.get("pdf_text", "")
                extracted_context = st.session_state.result.json()

                # Chat Model
                chat_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=api_key,
                    temperature=0.3
                )

                # Prompt
                system_instruction = f"""
                    You are Processly AI, an expert battery engineer assistant.

                    ### CONTEXT:
                    1. **VERIFIED METRICS:** {extracted_context}
                    2. **DOCUMENT CONTENT:** {raw_text[:30000]}

                    ### GOAL:
                    Answer the user's question accurately. Prioritize the 'VERIFIED METRICS' if the question is about specific values (like Voltage, Current). Use the 'DOCUMENT CONTENT' for general context or qualitative questions.
                    """

                messages_to_send = [
                    {"role": "system", "content": system_instruction},
                    *st.session_state.messages
                ]

                # Stream Response
                with chat_container.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    try:
                        for chunk in chat_llm.stream(messages_to_send):
                            full_response += chunk.content
                            response_placeholder.markdown(full_response)

                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    except Exception as e:
                        st.error(f"Chat Error: {e}")

elif not api_key:
    st.info("👈 Please enter your Google API Key in the sidebar to start Processly.")