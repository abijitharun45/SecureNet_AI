from fpdf import FPDF
import pandas as pd
from datetime import datetime
from typing import Optional


class ForensicReport(FPDF):
    """
    Custom PDF class for generating Network Forensic Reports.
    Inherits from FPDF to override header and footer methods.
    """

    def header(self):
        """Renders the professional header on every page."""
        # Set font for the company branding
        self.set_font('Arial', 'B', 15)

        # Title (Centered)
        # Cell(width, height, text, border, newline, align)
        self.cell(0, 10, 'SecureNet AI - Forensic Analysis Report', 0, 1, 'C')

        # Subtitle/Line break
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'Automated Deep Learning Threat Detection System', 0, 1, 'C')
        self.ln(5)

        # Horizontal line for separation
        self.line(10, 30, 200, 30)
        self.ln(10)

    def footer(self):
        """Renders the footer with page numbers."""
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def generate_threat_table(pdf: FPDF, df: pd.DataFrame):
    """
    Helper function to draw a styled table of critical threats.
    Optimized to reduce redundant DataFrame operations.
    """
    # Filter for threats only - optimized comparison
    threats = df[df['Detected_Type'].str.upper() != 'BENIGN']

    if threats.empty:
        pdf.set_font('Arial', 'I', 11)
        pdf.cell(0, 10, "No active threats identified in this log file.", 0, 1)
        return

    # Get Top 5 Threats by frequency
    top_threats = threats['Detected_Type'].value_counts().head(5)
    
    # Pre-calculate max confidence for all threats at once (vectorized)
    threat_confidence = threats.groupby('Detected_Type')['Confidence'].max()

    # Table Header Configuration
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(230, 230, 230)  # Light Gray background

    # Headers
    pdf.cell(90, 10, 'Attack Signature', 1, 0, 'L', 1)
    pdf.cell(50, 10, 'Frequency', 1, 0, 'C', 1)
    pdf.cell(50, 10, 'Max Confidence', 1, 1, 'C', 1)

    # Table Body
    pdf.set_font('Arial', '', 10)
    for attack_name, count in top_threats.items():
        # Use pre-calculated max confidence
        max_conf = threat_confidence[attack_name]

        pdf.cell(90, 10, str(attack_name), 1)
        pdf.cell(50, 10, str(count), 1, 0, 'C')
        pdf.cell(50, 10, f"{max_conf:.4f}", 1, 1, 'C')

    pdf.ln(5)


def create_pdf(df: pd.DataFrame) -> bytes:
    """
    Generates a full forensic PDF report from the analysis dataframe.
    Optimized to reduce redundant calculations.

    Args:
        df (pd.DataFrame): The dataframe containing 'Detected_Type' and 'Confidence' columns.

    Returns:
        bytes: The raw PDF bytes string (latin-1 encoded) suitable for download.
    """
    pdf = ForensicReport()
    pdf.add_page()

    # --- SECTION 1: METADATA ---
    pdf.set_font('Arial', '', 11)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Report Generated: {current_time}", 0, 1)
    pdf.ln(5)

    # --- SECTION 2: EXECUTIVE SUMMARY ---
    # Calculate statistics once - optimized
    total_packets = len(df)
    # Note: .str.upper() is necessary as model outputs may vary in case
    threat_mask = (df['Detected_Type'].str.upper() != 'BENIGN').values
    threat_count = int(threat_mask.sum())
    safe_count = total_packets - threat_count
    threat_percentage = (threat_count / total_packets) * 100 if total_packets > 0 else 0

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Executive Summary', 0, 1)

    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"Total Network Artifacts Scanned: {total_packets}", 0, 1)
    pdf.cell(0, 8, f"Benign Traffic Flows: {safe_count}", 0, 1)

    # Conditional formatting for threat count
    if threat_count > 0:
        pdf.set_text_color(194, 24, 7)  # Red color for alerts
        pdf.cell(0, 8, f"Malicious Flows Detected: {threat_count} ({threat_percentage:.2f}%)", 0, 1)
        pdf.set_text_color(0, 0, 0)  # Reset to black
    else:
        pdf.set_text_color(34, 139, 34)  # Green color
        pdf.cell(0, 8, "No Malicious Flows Detected (System Secure)", 0, 1)
        pdf.set_text_color(0, 0, 0)

    pdf.ln(10)

    # --- SECTION 3: THREAT INTELLIGENCE ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Critical Threat Analysis', 0, 1)

    generate_threat_table(pdf, df)
    pdf.ln(5)

    # --- SECTION 4: AI RECOMMENDATIONS ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Automated Mitigation Steps', 0, 1)
    pdf.set_font('Arial', '', 11)

    if threat_count > 0:
        recommendations = (
            "CRITICAL ALERT: Anomalous traffic patterns detected.\n"
            "1. Isolate the affected subnet immediately.\n"
            "2. Block source IPs associated with the identified attack signatures.\n"
            "3. Review firewall logs for the timestamp range specified in this report.\n"
            "4. Cross-reference payloads with CVE databases for specific exploits."
        )
    else:
        recommendations = (
            "System Status: NORMAL.\n"
            "1. Continue routine monitoring.\n"
            "2. No immediate action required.\n"
            "3. Schedule next periodic scan in 24 hours."
        )

    pdf.multi_cell(0, 8, recommendations)

    # Return PDF as bytes string
    return bytes(pdf.output(dest='S'))