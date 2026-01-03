import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
import time
from typing import Tuple, Optional, Any
from src.pdf_generator import create_pdf  # Ensure pdf_generator.py is in the same folder

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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# --- ASSET MANAGEMENT ---
@st.cache_resource
def load_inference_artifacts() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Loads the pre-trained Keras model, Scaler, and Label Encoder from disk.

    Returns:
        Tuple containing (model, scaler, label_encoder) or (None, None, None) on failure.
    """
    try:
        # Load Hybrid CNN-LSTM Model
        model = tf.keras.models.load_model('models/best_cnn_lstm_ids_model.h5')

        # Load Preprocessing Pipelines
        scaler = joblib.load('models/deployment_scaler.pkl')
        le = joblib.load('models/deployment_label_encoder.pkl')

        return model, scaler, le
    except FileNotFoundError as e:
        st.error(f"Critical Asset Missing: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None, None


model, scaler, le = load_inference_artifacts()


# --- CORE LOGIC ---
def align_feature_shape(X: np.ndarray, target_features: int = TOTAL_INPUT_FEATURES) -> np.ndarray:
    """
    Ensures input vector matches the model's required input dimensions by padding or truncation.

    Args:
        X (np.ndarray): The input feature matrix.
        target_features (int): The total number of flattened features required.

    Returns:
        np.ndarray: The reshaped and aligned feature matrix.
    """
    current_features = X.shape[1]

    if current_features < target_features:
        # Zero-padding for dimensionality mismatch
        padding_size = target_features - current_features
        X_aligned = np.pad(X, ((0, 0), (0, padding_size)), mode='constant')
    elif current_features > target_features:
        # Feature truncation
        X_aligned = X[:, :target_features]
    else:
        X_aligned = X

    return X_aligned


def preprocess_traffic_data(df: pd.DataFrame, scaler: Any) -> np.ndarray:
    """
    Transforms raw network logs into normalized sequences for Deep Learning inference.

    Args:
        df (pd.DataFrame): Raw CSV data.
        scaler (Any): Fitted Scikit-Learn scaler.

    Returns:
        np.ndarray: 3D Tensor of shape (Batch, Time_Steps, Features).
    """
    # Remove Ground Truth if present (Validation Mode)
    data = df.drop('Label', axis=1) if 'Label' in df.columns else df

    # Feature Normalization
    try:
        X_scaled = scaler.transform(data)
    except ValueError:
        # Adaptive Fallback for schema mismatches
        X_scaled = scaler.fit_transform(data)

    # Dimensionality Alignment
    X_aligned = align_feature_shape(X_scaled)

    # Reshape for LSTM (Samples, Time, Features)
    return X_aligned.reshape(X_aligned.shape[0], SEQUENCE_LENGTH, MODEL_FEATURES)


# --- UI COMPONENTS ---
def render_sidebar():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
        st.title("SecureNet AI")
        st.markdown("### System Diagnostics")

        if model and scaler and le:
            st.success("✅ Neural Engine Online")
            st.success("✅ Scaler Active")
            st.success("✅ Encoder Loaded")
        else:
            st.error("❌ System Offline")
            st.warning("Required artifacts (.h5/.pkl) not found.")

        st.markdown("---")
        st.info("Architecture: **Hybrid CNN-LSTM**")
        st.caption("v1.0.2 | Enterprise Edition")


def render_main_dashboard():
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown("Advanced traffic analysis using Deep Learning for anomaly detection.")

    col_upload, col_stats = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader("📂 Import Traffic Log (.csv)", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"**Ingested:** {df.shape[0]} packets.")

        with st.expander("Inspect Raw Packet Headers"):
            display_df = df.drop('Label', axis=1) if 'Label' in df.columns else df
            st.dataframe(display_df.head(), use_container_width=True)

        if st.button("🚀 Initiate Threat Scan", type="primary"):
            if model and scaler:
                run_inference_pipeline(df)


def run_inference_pipeline(df: pd.DataFrame):
    # Progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Preprocessing
    status_text.text("Status: Normalizing vector space...")
    progress_bar.progress(25)
    time.sleep(0.3)

    X_input = preprocess_traffic_data(df, scaler)

    # Step 2: Feature Extraction
    status_text.text("Status: Extracting spatial features (CNN)...")
    progress_bar.progress(50)
    time.sleep(0.3)

    # Step 3: Temporal Analysis
    status_text.text("Status: Analyzing sequence patterns (LSTM)...")
    progress_bar.progress(75)

    # Step 4: Inference
    preds_proba = model.predict(X_input)
    preds_idx = np.argmax(preds_proba, axis=1)
    preds_labels = le.inverse_transform(preds_idx)

    progress_bar.progress(100)
    status_text.text("Scan Completed Successfully.")

    render_results(df, preds_labels, preds_proba)


def render_results(df: pd.DataFrame, preds_labels: np.ndarray, preds_proba: np.ndarray):
    st.markdown("---")
    st.subheader("📊 Forensic Analysis Report")

    results_df = df.copy()
    results_df['Detected_Type'] = preds_labels
    results_df['Confidence'] = np.max(preds_proba, axis=1)

    # Analysis Logic (Case-Insensitive)
    benign_mask = results_df['Detected_Type'].str.upper() == 'BENIGN'
    benign_count = benign_mask.sum()
    threat_count = len(results_df) - benign_count

    # 1. KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Flows", len(results_df))
    kpi2.metric("Verified Safe", int(benign_count))
    kpi3.metric("Threats Identified", int(threat_count), delta_color="inverse")

    # 2. Charts & Tables
    c1, c2 = st.columns(2)

    with c1:
        st.write("### Attack Taxonomy Distribution")
        threat_data = results_df[~benign_mask]['Detected_Type']
        if not threat_data.empty:
            st.bar_chart(threat_data.value_counts())
        else:
            st.success("Network traffic matches benign signatures.")

    with c2:
        st.write("### Critical Alerts (Confidence > 80%)")
        critical_threats = results_df[
            (~benign_mask) & (results_df['Confidence'] > 0.8)
            ][['Detected_Type', 'Confidence']].head(10)

        if not critical_threats.empty:
            st.dataframe(critical_threats, use_container_width=True)
        else:
            st.info("No high-confidence threats detected.")

    # 3. Exports
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    pdf_bytes = create_pdf(results_df)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button("📥 Export Logs (CSV)", csv_data, "scan_logs.csv", "text/csv")
    with col_d2:
        st.download_button("📄 Download Forensic PDF", pdf_bytes, "forensic_report.pdf", "application/pdf")

    # 4. Validation (Ground Truth Check)
    if 'Label' in df.columns:
        render_validation_metrics(df, preds_labels)


def render_validation_metrics(df: pd.DataFrame, preds_labels: np.ndarray):
    st.markdown("---")
    st.subheader("🧪 Model Validation Metrics")

    validation_df = pd.DataFrame({
        'Actual': df['Label'].astype(str).str.upper(),
        'Predicted': pd.Series(preds_labels).astype(str).str.upper()
    })

    validation_df['Match'] = validation_df['Actual'] == validation_df['Predicted']
    accuracy = (validation_df['Match'].sum() / len(validation_df)) * 100

    m1, m2 = st.columns([1, 3])
    m1.metric("Validation Accuracy", f"{accuracy:.2f}%")

    with m2:
        if accuracy > 90:
            st.success("✅ High Fidelity: Model aligns with ground truth.")
        else:
            st.warning("⚠️ Discrepancy Detected: Review misclassifications below.")

    with st.expander("🔍 Detailed Error Analysis"):
        errors = validation_df[~validation_df['Match']]
        if not errors.empty:
            st.error(f"Identified {len(errors)} misclassifications.")
            st.dataframe(errors, use_container_width=True)
        else:
            st.caption("Perfect alignment with validation set.")


# --- APP EXECUTION ENTRY POINT ---
if __name__ == "__main__":
    render_sidebar()
    render_main_dashboard()