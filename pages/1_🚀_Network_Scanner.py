import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import time
import altair as alt
from typing import Tuple, Optional, Any
from src.pdf_generator import create_pdf
from src.ai_analyst import GroqAnalyst

# --- CONSTANTS & CONFIGURATION ---
SEQUENCE_LENGTH = 10
MODEL_FEATURES = 4
TOTAL_INPUT_FEATURES = SEQUENCE_LENGTH * MODEL_FEATURES  # 40

st.set_page_config(
    page_title="SecureNet AI | Intelligent IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD CYBER-OPS THEME ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("assets/style.css")
except FileNotFoundError:
    pass

# --- ASSET MANAGEMENT ---
@st.cache_resource
def load_inference_artifacts() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    try:
        model = tf.keras.models.load_model('models/best_cnn_lstm_ids_model.h5')
        scaler = joblib.load('models/deployment_scaler.pkl')
        le = joblib.load('models/deployment_label_encoder.pkl')
        return model, scaler, le
    except Exception as e:
        st.error(f"Critical Asset Missing: {e}")
        return None, None, None

model, scaler, le = load_inference_artifacts()
analyst = GroqAnalyst()  # Initialize AI Analyst

# --- CORE LOGIC ---
def align_feature_shape(X: np.ndarray, target_features: int = TOTAL_INPUT_FEATURES) -> np.ndarray:
    current_features = X.shape[1]
    if current_features < target_features:
        padding_size = target_features - current_features
        X_aligned = np.pad(X, ((0, 0), (0, padding_size)), mode='constant')
    elif current_features > target_features:
        X_aligned = X[:, :target_features]
    else:
        X_aligned = X
    return X_aligned

def preprocess_traffic_data(df: pd.DataFrame, scaler: Any) -> np.ndarray:
    data = df.drop('Label', axis=1) if 'Label' in df.columns else df
    try:
        X_scaled = scaler.transform(data)
    except ValueError:
        X_scaled = scaler.fit_transform(data)
    
    X_aligned = align_feature_shape(X_scaled)
    return X_aligned.reshape(X_aligned.shape[0], SEQUENCE_LENGTH, MODEL_FEATURES)

# --- BATCH SIMULATOR LOGIC (VECTORIZED) ---
def run_live_simulation(df_source):
    st.markdown("### ⚡ HIGH-VELOCITY TRAFFIC ANALYSIS")
    
    # 1. Configuration
    limit = st.slider("Select Packet Batch Size", min_value=100, max_value=5000, value=1000)
    
    # 2. Key Action: Execute Scan
    if st.button("🚀 EXECUTE BATCH SCAN", type="primary"):
        with st.spinner(f"Processing {limit} packets in neural engine..."):
            # Load Batch
            if len(df_source) > limit:
                batch_df = df_source.sample(limit)
            else:
                batch_df = df_source
            
            # Vectorized Processing
            start_time = time.time()
            X_input = preprocess_traffic_data(batch_df, scaler)
            preds_proba = model.predict(X_input, verbose=0)
            
            # Post-Processing
            pred_idx = np.argmax(preds_proba, axis=1)
            confidence_scores = np.max(preds_proba, axis=1)
            pred_labels = le.inverse_transform(pred_idx)
            duration = time.time() - start_time
            
            # Store Results in Session State
            results_df = batch_df.copy()
            results_df['Detected_Type'] = pred_labels
            results_df['Confidence'] = confidence_scores
            results_df['Is_Threat'] = results_df['Detected_Type'].str.upper() != 'BENIGN'
            
            st.session_state['scan_results'] = results_df
            
            # Calculate metrics FIRST
            total_packets = len(results_df)
            threat_count = results_df['Is_Threat'].sum()
            benign_count = total_packets - threat_count
            
            st.session_state['scan_metrics'] = {
                'velocity': total_packets / duration if duration > 0 else 0,
                'duration': duration,
                'total': total_packets,
                'threats': threat_count,
                'benign': benign_count
            }
            
            # Clear previous AI report if new scan is run
            if 'ai_report' in st.session_state:
                del st.session_state['ai_report']

    # 3. Display Results (Persisted)
    if 'scan_results' in st.session_state:
        results_df = st.session_state['scan_results']
        metrics = st.session_state['scan_metrics']
        
        # Metrics Display
        k1, k2, k3 = st.columns(3)
        k1.metric("SCAN VELOCITY", f"{metrics['velocity']:.0f} pkts/sec", delta=f"{metrics['duration']:.2f}s total")
        k2.metric("THREATS DETECTED", int(metrics['threats']), delta_color="inverse")
        k3.metric("SAFE TRAFFIC", int(metrics['benign']))
        
        # Visualization
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### 📉 Confidence Stream")
            # Downsample for visualization
            chart_df = results_df.reset_index(drop=True).reset_index()
            if len(chart_df) > 500:
                chart_df = chart_df.sample(500).sort_values("index")
            
            chart = alt.Chart(chart_df).mark_circle(size=60).encode(
                x=alt.X('index', title='Packet Sequence'),
                y=alt.Y('Confidence', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('Is_Threat', scale=alt.Scale(domain=[True, False], range=['#ff0000', '#00ff41']), legend=None),
                tooltip=['Detected_Type', 'Confidence']
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)

        with c2:
            st.markdown("#### 🚨 Top Threats")
            if metrics['threats'] > 0:
                top_threats = results_df[results_df['Is_Threat']]['Detected_Type'].value_counts().head(5)
                st.dataframe(top_threats, use_container_width=True)
            else:
                st.success("No threats found.")

        # AI Analyst Section
        if metrics['threats'] > 0:
            st.markdown("---")
            st.subheader("🧠 Neural Forensic Analyst")
            
            # Select threat to analyze
            target_threat = st.selectbox("Select Threat Class", results_df[results_df['Is_Threat']]['Detected_Type'].unique())
            
            if st.button("Analyze with Groq AI"):
                with st.spinner("Querying Neural Defense Grid..."):
                    # Find a sample packet
                    sample_pkt = results_df[results_df['Detected_Type'] == target_threat].iloc[0]
                    # Call API
                    report = analyst.analyze_threat(target_threat, sample_pkt['Confidence'], sample_pkt.to_string())
                    st.session_state['ai_report'] = report
            
            # Persist AI Report
            if 'ai_report' in st.session_state:
                st.info("📝 Analysis Generated")
                st.markdown(st.session_state['ai_report'])

        # Export Area
        st.markdown("### 📥 Evidence Locker")
        col_d1, col_d2 = st.columns(2)
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        # Re-generate PDF on fly or cache it? Re-generating is cheap enough for now but ensuring button doesn't reset state is key.
        # Streamlit buttons reload the script, but session_state will keep the results_df alive, so create_pdf works.
        pdf_bytes = create_pdf(results_df)
        
        with col_d1:
            st.download_button("💾 Save Full Log (CSV)", csv, "batch_scan.csv", "text/csv")
        with col_d2:
            st.download_button("📄 Print Forensic Report (PDF)", pdf_bytes, "report.pdf", "application/pdf")

def run_static_analysis(df):
    if st.button("🚀 EXECUTE SCAN", type="primary", key="static_scan_btn"): 
        with st.spinner("Analyzing uploaded traffic log..."):
            # Inference
            X_input = preprocess_traffic_data(df, scaler)
            preds_proba = model.predict(X_input)
            preds_idx = np.argmax(preds_proba, axis=1)
            preds_labels = le.inverse_transform(preds_idx)
            
            # Results Construction
            results_df = df.copy()
            results_df['Detected_Type'] = preds_labels
            results_df['Confidence'] = np.max(preds_proba, axis=1)
            results_df['Is_Threat'] = results_df['Detected_Type'].str.upper() != 'BENIGN'
            
            # Save to Session State
            st.session_state['static_results'] = results_df
            
            # Clear previous static report
            if 'static_ai_report' in st.session_state:
                del st.session_state['static_ai_report']

    # Display Persisted Results
    if 'static_results' in st.session_state:
        results_df = st.session_state['static_results']
        
        st.markdown("### 📊 SCAN RESULTS")
        
        # Summary Metrics
        total = len(results_df)
        threats = results_df['Is_Threat'].sum()
        
        m1, m2 = st.columns(2)
        m1.metric("TOTAL RECORDS", total)
        m2.metric("THREATS DETECTED", int(threats), delta_color="inverse")

        st.dataframe(results_df[['Detected_Type', 'Confidence']], use_container_width=True)
        
        # AI Analyst for Static
        if threats > 0:
            st.markdown("---")
            st.subheader("🧠 Neural Forensic Analyst")
            target = st.selectbox("Select Threat to Analyze", results_df[results_df['Is_Threat']]['Detected_Type'].unique(), key="static_ai_select")
            
            if st.button("Analyze Threat (Static)", key="static_ai_btn"):
                with st.spinner(" analyzing..."):
                    sample = results_df[results_df['Detected_Type'] == target].iloc[0]
                    report = analyst.analyze_threat(target, sample['Confidence'], sample.to_string())
                    st.session_state['static_ai_report'] = report
            
            if 'static_ai_report' in st.session_state:
                st.markdown(st.session_state['static_ai_report'])
        
        # Export
        st.markdown("### 📥 Evidence Locker")
        csv = results_df.to_csv(index=False).encode('utf-8')
        pdf_bytes = create_pdf(results_df)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📥 DOWNLOAD LOGS", csv, "scan_results.csv", "text/csv")
        with col_d2:
            st.download_button("📄 Download Forensic PDF", pdf_bytes, "forensic_report.pdf", "application/pdf")


# --- UI COMPONENTS ---
def render_sidebar():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
        st.title("SecureNet AI")
        st.markdown("### ⚙️ SCANNER CONFIG")
        
        mode = st.radio("Scanning Mode", ["📂 Static File Upload", "🔴 Live Traffic Simulator"])
        
        st.markdown("---")
        st.info(f"AI Analyst: {'🟢 ONLINE' if analyst.client else '🟠 OFFLINE (Fallback Mode)'}")

        return mode

def render_main_dashboard(mode):
    st.markdown('<div class="cyber-header"><h1>🚀 Network Scanner</h1></div>', unsafe_allow_html=True)
    
    if mode == "📂 Static File Upload":
        uploaded_file = st.file_uploader("Upload Traffic Log (.csv)", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            run_static_analysis(df) # Call directly without button
                
    elif mode == "🔴 Live Traffic Simulator":
        st.info("ℹ️ Simulating real-time network traffic from test dataset.")
        
        # Load data once here
        try:
             df_test = pd.read_csv("data/test_mini.csv") 
             run_live_simulation(df_test)
        except Exception as e:
             st.error(f"Data Load Error: {e}")


# --- APP EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    mode = render_sidebar()
    render_main_dashboard(mode)
