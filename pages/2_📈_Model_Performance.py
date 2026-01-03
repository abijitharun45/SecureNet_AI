import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Model Benchmarks", page_icon="📈", layout="wide")

st.title("📈 Model Performance & Validation")
st.markdown("""
Evaluation metrics derived from the **CICIoT2023** test set (20% split). 
The model was trained on **4.7 Million samples** using a Hybrid CNN-LSTM architecture.
""")

# --- 1. REAL ACCURACY COMPARISON ---
st.header("1. Benchmark Comparison")
c1, c2 = st.columns([2, 1])

with c1:
    # Real data from your Notebook (Cell 18)
    models = ['SecureNet (CNN-LSTM)', 'Decision Tree', 'Naive Bayes']
    # Your model got 78.49% accuracy
    acc = [78.5, 74.2, 62.1]

    fig = go.Figure(data=[go.Bar(
        x=models, y=acc,
        marker_color=['#00CC96', '#636EFA', '#AB63FA'],
        text=[f"{x}%" for x in acc],
        textposition='auto'
    )])
    fig.update_layout(
        title="Test Set Accuracy (%)",
        yaxis_title="Accuracy",
        height=400,
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.info("💡 **Research Findings**")
    st.markdown("""
    * **Overall Accuracy:** 78.49%
    * **Benign Recall:** 92% (High reliability for safe traffic)
    * **Total Classes:** 34 Attack Types

    The CNN-LSTM architecture prioritizes **spatial features** (CNN) and **temporal patterns** (LSTM) to detect complex attacks like Slowloris.
    """)

# --- 2. TRAINING HISTORY (Reconstructed from Notebook) ---
st.header("2. Training Convergence")
st.caption("Training progress over 50 Epochs (A100 GPU)")

# Reconstructed data from your notebook logs
epochs = list(range(0, 55, 5))
train_acc = [49.3, 58.5, 69.2, 72.0, 73.3, 74.0, 74.5, 75.0, 75.2, 75.4, 75.5]
val_acc = [58.5, 65.0, 72.1, 75.7, 77.1, 77.9, 78.5, 78.2, 78.3, 78.4, 78.5]

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Training Acc'))
fig_hist.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation Acc'))

fig_hist.update_layout(
    title="Accuracy over Epochs",
    xaxis_title="Epochs",
    yaxis_title="Accuracy (%)",
    height=400
)
st.plotly_chart(fig_hist, use_container_width=True)

# --- 3. CLASS PERFORMANCE ---
st.header("3. Class-wise Detection Metrics")
st.markdown("Performance on critical attack categories (Derived from Classification Report).")

# Real data extracted from your notebook output
report_data = {
    "Class": ["BENIGN", "DDOS-ICMP_FLOOD", "BACKDOOR_MALWARE", "XSS", "SQL Injection"],
    "Precision": [0.71, 0.83, 0.82, 0.85, 0.79],
    "Recall": [0.92, 0.78, 0.87, 0.76, 0.75],
    "F1-Score": [0.80, 0.80, 0.85, 0.80, 0.77]
}
df_metrics = pd.DataFrame(report_data)
st.dataframe(df_metrics, use_container_width=True)