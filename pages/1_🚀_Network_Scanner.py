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
@st.cache_data
def load_css(file_name):
    """Load CSS with caching to avoid repeated file I/O."""
    try:
        with open(file_name) as f:
            return f.read()
    except FileNotFoundError:
        return None

css_content = load_css("assets/style.css")
if css_content:
    st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)

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

@st.cache_data
def load_test_data(file_path: str) -> pd.DataFrame:
    """Load test data with caching to avoid repeated I/O."""
    return pd.read_csv(file_path)

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
            # Load Batch - optimized sampling
            if len(df_source) > limit:
                # Use numpy random choice for faster sampling
                sample_indices = np.random.choice(len(df_source), size=limit, replace=False)
                batch_df = df_source.iloc[sample_indices].copy()
            else:
                batch_df = df_source.copy()
            
            # Vectorized Processing
            start_time = time.time()
            X_input = preprocess_traffic_data(batch_df, scaler)
            preds_proba = model.predict(X_input, verbose=0)
            
            # Post-Processing - optimized with numpy operations
            pred_idx = np.argmax(preds_proba, axis=1)
            confidence_scores = np.max(preds_proba, axis=1)
            pred_labels = le.inverse_transform(pred_idx)
            duration = time.time() - start_time
            
            # Store Results in Session State
            results_df = batch_df.copy()
            results_df['Detected_Type'] = pred_labels
            results_df['Confidence'] = confidence_scores
            # Optimize string comparison using vectorized operation
            results_df['Is_Threat'] = (results_df['Detected_Type'].str.upper() != 'BENIGN').values

            # --- BONUS: GLOBAL THREAT MAP DATA ---
            # (Map is now a static GIF for better aesthetics)
            
            st.session_state['scan_results'] = results_df
            
            # Calculate metrics using numpy for better performance
            total_packets = len(results_df)
            threat_count = int(results_df['Is_Threat'].sum())
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
        
        # --- GLOBAL THREAT MAP ---
        st.markdown("### 🗺️ LIVE GLOBAL THREAT TRACE")
        
        # 1. Map Selection Dropdown (Hidden power feature for you)
        map_choice = st.selectbox(
            "Select Intelligence Feed Source:",
            ["Radware (Tactical Dark)", "SonicWall (High-Data)", "Fortinet (3D Globe)"],
            label_visibility="collapsed" # Hides the label to keep it clean
        )

        try:
            import streamlit.components.v1 as components
            
            # 2. Define the URLs for the maps you liked
            map_urls = {
                "Radware (Tactical Dark)": "https://livethreatmap.radware.com/",
                "SonicWall (High-Data)": "https://attackmap.sonicwall.com/live-attack-map/",
                "Fortinet (3D Globe)": "https://threatmap.fortiguard.com/"
            }
            
            selected_url = map_urls[map_choice]

            # 3. Render the Map in a nice clear window
            # We use a slightly taller height (650px) to accommodate the flat maps better
            components.iframe(
                src=selected_url,
                height=650,
                scrolling=True
            )
            
            # Dynamic Caption based on selection
            st.caption(f"🔴 Live Feed: {map_choice} • Real-Time Attack Vector Analysis")

        except Exception as e:
            # Fallback text if internet/iframe fails
            st.warning("⚠️ Live Map Feed Unavailable (Check Internet Connection)")
        st.markdown("---")

        # Visualization
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("#### 📉 Confidence Stream")
            # Downsample for visualization - optimized
            chart_df = results_df[['Confidence', 'Is_Threat', 'Detected_Type']].reset_index(drop=True).reset_index()
            if len(chart_df) > 500:
                # Use systematic sampling for better distribution
                step = len(chart_df) // 500
                chart_df = chart_df.iloc[::step].copy()
            
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
                # Optimized: filter once and reuse
                threat_df = results_df[results_df['Is_Threat']]
                top_threats = threat_df['Detected_Type'].value_counts().head(5)
                st.dataframe(top_threats, use_container_width=True)
            else:
                st.success("No threats found.")

        # AI Analyst Section
        if metrics['threats'] > 0:
            st.markdown("---")
            st.subheader("🧠 Neural Forensic Analyst")
            
            # Select threat to analyze - reuse filtered threat_df
            threat_df = results_df[results_df['Is_Threat']]
            unique_threats = threat_df['Detected_Type'].unique()
            target_threat = st.selectbox("Select Threat Class", unique_threats)
            
            if st.button("Analyze with Groq AI"):
                with st.spinner("Querying Neural Defense Grid..."):
                    # Find a sample packet - optimized query
                    sample_pkt = threat_df[threat_df['Detected_Type'] == target_threat].iloc[0]
                    # Call API with minimal data
                    report = analyst.analyze_threat(target_threat, sample_pkt['Confidence'], sample_pkt.to_string())
                    st.session_state['ai_report'] = report
            
            # Persist AI Report
            if 'ai_report' in st.session_state:
                st.info("📝 Analysis Generated")
                st.markdown(st.session_state['ai_report'])

        # Export Area
        st.markdown("### 📥 Evidence Locker")
        col_d1, col_d2 = st.columns(2)
        
        # Pre-generate export data once (using session state to cache)
        if 'export_csv' not in st.session_state or st.session_state.get('last_scan_id') != id(results_df):
            st.session_state['export_csv'] = results_df.to_csv(index=False).encode('utf-8')
            st.session_state['export_pdf'] = create_pdf(results_df)
            st.session_state['last_scan_id'] = id(results_df)
        
        with col_d1:
            st.download_button("💾 Save Full Log (CSV)", st.session_state['export_csv'], "batch_scan.csv", "text/csv")
        with col_d2:
            st.download_button("📄 Print Forensic Report (PDF)", st.session_state['export_pdf'], "report.pdf", "application/pdf")

def run_static_analysis(df):
    if st.button("🚀 EXECUTE SCAN", type="primary", key="static_scan_btn"): 
        with st.spinner("Analyzing uploaded traffic log..."):
            # Inference
            X_input = preprocess_traffic_data(df, scaler)
            preds_proba = model.predict(X_input, verbose=0)
            preds_idx = np.argmax(preds_proba, axis=1)
            preds_labels = le.inverse_transform(preds_idx)
            
            # Results Construction - optimized
            results_df = df.copy()
            results_df['Detected_Type'] = preds_labels
            results_df['Confidence'] = np.max(preds_proba, axis=1)
            # Optimize string comparison using vectorized operation
            results_df['Is_Threat'] = (results_df['Detected_Type'].str.upper() != 'BENIGN').values
            
            # Save to Session State
            st.session_state['static_results'] = results_df
            
            # Clear previous static report and exports
            if 'static_ai_report' in st.session_state:
                del st.session_state['static_ai_report']
            if 'static_export_csv' in st.session_state:
                del st.session_state['static_export_csv']
            if 'static_export_pdf' in st.session_state:
                del st.session_state['static_export_pdf']

    # Display Persisted Results
    if 'static_results' in st.session_state:
        results_df = st.session_state['static_results']
        
        st.markdown("### 📊 SCAN RESULTS")
        
        # Summary Metrics - optimized calculation
        total = len(results_df)
        threats = int(results_df['Is_Threat'].sum())
        
        m1, m2 = st.columns(2)
        m1.metric("TOTAL RECORDS", total)
        m2.metric("THREATS DETECTED", threats, delta_color="inverse")

        st.dataframe(results_df[['Detected_Type', 'Confidence']], use_container_width=True)
        
        # AI Analyst for Static
        if threats > 0:
            st.markdown("---")
            st.subheader("🧠 Neural Forensic Analyst")
            threat_df = results_df[results_df['Is_Threat']]
            target = st.selectbox("Select Threat to Analyze", threat_df['Detected_Type'].unique(), key="static_ai_select")
            
            if st.button("Analyze Threat (Static)", key="static_ai_btn"):
                with st.spinner(" analyzing..."):
                    sample = threat_df[threat_df['Detected_Type'] == target].iloc[0]
                    report = analyst.analyze_threat(target, sample['Confidence'], sample.to_string())
                    st.session_state['static_ai_report'] = report
            
            if 'static_ai_report' in st.session_state:
                st.markdown(st.session_state['static_ai_report'])
        
        # Export - optimized with caching
        st.markdown("### 📥 Evidence Locker")
        
        # Generate export data once and cache in session state
        if 'static_export_csv' not in st.session_state:
            st.session_state['static_export_csv'] = results_df.to_csv(index=False).encode('utf-8')
            st.session_state['static_export_pdf'] = create_pdf(results_df)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📥 DOWNLOAD LOGS", st.session_state['static_export_csv'], "scan_results.csv", "text/csv")
        with col_d2:
            st.download_button("📄 Download Forensic PDF", st.session_state['static_export_pdf'], "forensic_report.pdf", "application/pdf")


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
        
        # Load data once here with caching
        try:
             df_test = load_test_data("data/test_mini.csv")
             run_live_simulation(df_test)
        except Exception as e:
             st.error(f"Data Load Error: {e}")


# --- APP EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    mode = render_sidebar()
    render_main_dashboard(mode)
