
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="SecureNet AI | Enterprise Security",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.title("SecureNet AI")
    st.info("v1.0.2 | Enterprise Edition")

# Hero Section
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🛡️ SecureNet AI")
    st.markdown("### Next-Generation Network Forensic Intelligence")
    st.markdown("""
    **SecureNet AI** bridges the gap between traditional firewalls and modern Deep Learning. 
    By utilizing a **Hybrid CNN-LSTM Architecture**, this system detects complex, 
    low-frequency attacks that rule-based systems miss.

    #### 🚀 Key Capabilities:
    * **Deep Packet Inspection:** Analyzes 40+ traffic features per flow.
    * **Temporal Analysis:** LSTM layers detect slow-pattern attacks (e.g., Slowloris).
    * **Automated Forensics:** Generates instant PDF reports for compliance.
    """)

    st.success("👈 Select **'Network Scanner'** in the sidebar to start.")

with col2:
    st.markdown("### ⚡ System Stats")
    st.metric(label="Model Accuracy", value="78.5%", delta="Valid. Score")
    st.metric(label="Benign Traffic Recall", value="92.0%", delta="High Reliability")
    st.metric(label="Threat Classes", value="34 Types")
