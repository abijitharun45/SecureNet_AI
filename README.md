# 🛡️ SecureNet AI: Hybrid CNN-LSTM Intrusion Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-green)

**SecureNet AI** is a next-generation Network Intrusion Detection System (NIDS) designed for IoT environments. It utilizes a **Hybrid Deep Learning Architecture** combining **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Long Short-Term Memory (LSTM)** networks for temporal sequence analysis.

---

## 🚀 Key Features
* **Deep Packet Inspection:** Analyzes **40+ traffic features** per flow.
* **Cyber-Ops Dashboard:** Futuristic dark/neon UI with **Vectorized Batch Processing** for scanning 1000+ packets/sec.
* **AI Security Analyst:** Integrated **Groq (Llama 3.3 70B)** for real-time forensic reports and threat mitigation.
* **Persistent Forensics:** State-managed analysis ensuring reports and logs persist during forensic deep-dives.

---

## 🔬 Scientific Methodology
This system was trained on the massive **CIC-IoT-2023** dataset (43 Million records) using a rigorous data engineering pipeline.

### Data Balancing Strategy
To address extreme class imbalance, a **Hybrid Sampling Strategy** was applied:
* **Benign Traffic:** 100% retention (1.05M samples) to maintain high False Positive reliability.
* **Rare Attacks (<100k):** Upsampled to 250,000 using **SMOTE/ADASYN** (Synthetic Minority Over-sampling).
* **Major Attacks (>100k):** Downsampled to 80,000 to prevent model bias.

### Model Architecture (CNN-LSTM)
* **Input:** Sequence of 10 Packets × 4 Features.
* **CNN Layers:** Extract spatial correlations between packet headers.
* **LSTM Layers:** Capture time-series dependencies (flow duration, inter-arrival time).
* **Output:** Softmax classification for **34 distinct attack classes**.

---

## 📊 Performance Metrics
Evaluated on a 20% independent test set (1.1M samples):

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Accuracy** | **78.49%** | High for 34-class classification |
| **Benign Recall** | **92.00%** | Critical for enterprise usability (Low False Positives) |
| **Precision (Avg)** | **81.00%** | Reliable threat identification |

---

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.10+
* TensorFlow (GPU recommended for retraining)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/abijitharun45/SecureNet_AI.git
cd SecureNet_AI

# Install dependencies
pip install -r requirements.txt

# Configure Secrets
# Create a file .streamlit/secrets.toml and add your Groq API key:
# GROQ_API_KEY = "gsk_..."

# Run the dashboard
streamlit run Home.py
```
## 📂 Project Structure
```bash
📂 SecureNet_AI/
├── 📜 Home.py                     # Main Landing Page
├── 📜 requirements.txt            # Project Dependencies
├── 📂 pages/
│   ├── 1_🚀_Network_Scanner.py    # Detection Engine Interface
│   └── 2_📈_Model_Performance.py  # Benchmarking Module
├── 📂 src/
│   ├── training_pipeline.py       # Modular training logic
│   └── pdf_generator.py           # Reporting engine
├── 📂 models/
│   ├── best_cnn_lstm_ids_model.h5 # Trained CNN-LSTM Model
│   ├── deployment_scaler.pkl      # Feature Scaler
│   └── deployment_label_encoder.pkl # Label Encoder
└── 📂 data/
    ├── CICIoT2023_balanced_test.csv # Full Test Dataset
    └── test_mini.csv              # Mini Sample for Upload