import streamlit as st
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SecureNet AI | Cyber-Ops",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CSS ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("assets/style.css")
except FileNotFoundError:
    st.error("⚠️ CSS File not found. UI might look standard.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=100)
    st.markdown("## OPERATOR: ADMIN")
    st.markdown("---")
    st.markdown("### 📡 SYSTEM STATUS")
    st.markdown("`CORE:` **ONLINE**", unsafe_allow_html=True)
    st.markdown("`AI ENGINE:` **ACTIVE**", unsafe_allow_html=True)
    st.markdown("`LATENCY:` **24ms**", unsafe_allow_html=True)
    st.markdown("---")
    st.info("v2.0.0 | CYBER-OPS EDITION")

# --- HERO SECTION ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="cyber-header"><h1>🛡️ SecureNet AI</h1></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cyber-card">
        <h3>>> NEXT-GEN INTRUSION DETECTION SYSTEM</h3>
        <p>Initializing neural defense protocols...</p>
        <p>Combining <b>Convolutional Neural Networks</b> with <b>Long Short-Term Memory</b> units to detect zero-day anomalies in real-time.</p>
        <br>
        <ul>
            <li>✨ <b>Live Packet Stream:</b> Real-time traffic monitoring</li>
            <li>🧠 <b>AI Forensic Analyst:</b> Automated threat intelligence</li>
            <li>⚡ <b>Hybrid Architecture:</b> CNN-LSTM based engine</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.success("✅ SYSTEM READY. Initialize 'Network Scanner' to begin operations.")

with col2:
    st.markdown("### ⚡ MODEL SPECS")
    st.metric(label="TRAINING DATA", value="5.9M", delta="Balanced Samples")
    st.metric(label="ACCURACY", value="78.5%", delta="Validation Set")
    st.metric(label="CLASSES", value="34", delta="Attack Types")

# Footer
st.markdown("---")
st.markdown("<center>SECURENET AI © 2024 | AUTHORIZED PERSONNEL ONLY</center>", unsafe_allow_html=True)
