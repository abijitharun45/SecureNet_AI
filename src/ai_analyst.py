import os
import streamlit as st
from groq import Groq

# Constants

class GroqAnalyst:
    def __init__(self, api_key=None):
        # Try to get API key from various sources
        self.api_key = api_key or os.getenv("GROQ_API_KEY") 
        if not self.api_key and "GROQ_API_KEY" in st.secrets:
            self.api_key = st.secrets["GROQ_API_KEY"]
            
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def analyze_threat(self, attack_type, confidence, packet_data=None):
        """
        Generates a forensic analysis of the detected threat.
        """
        if not self.client:
            return self._fallback_analysis(attack_type)

        try:
            prompt = self._construct_prompt(attack_type, confidence, packet_data)
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Tier-3 Cybersecurity Analyst at a clear, military-grade Security Operations Center (SOC). Your output must be concise, technical, and actionable. Use markdown. No fluff."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,  # Optimal for security analysis: precise, consistent, no hallucination
                max_tokens=1024,  # Full detailed forensic reports
            )
            
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"⚠️ **AI Analyst Offline:** {str(e)}\n\n" + self._fallback_analysis(attack_type)

    def _construct_prompt(self, attack_type, confidence, packet_data):
        # Full packet data sent for maximum analysis quality
        if packet_data:
            packet_summary = str(packet_data)
        else:
            packet_summary = "Traffic signature analysis confirmed anomalous behavior."
        
        return f"""
        **ALERT:** Network Intrusion Detected
        **THREAT:** {attack_type}
        **CONFIDENCE:** {confidence:.1%}
        
        **Full Packet Context:**
        {packet_summary}

        Provide a comprehensive structured forensic report:
        1. **Attack Vector:** What is this attack and how does it work?
        2. **Risk Assessment:** Severity level (Critical/High/Medium) and potential impact on the network.
        3. **Indicators of Compromise (IoCs):** Key signatures to watch for.
        4. **Immediate Mitigation:** 3-5 specific, actionable steps (e.g., Block UDP port 53, Rate limit SYN packets, isolate affected host).
        5. **Long-Term Recommendations:** Preventive measures to avoid recurrence.
        """

    def _fallback_analysis(self, attack_type):
        """
        Hardcoded expert rules for demo reliability when API fails.
        """
        reports = {
            "DDoS-ICMP_FLOOD": """
            **⚠️ EMERGENCY: ICMP Flood Detected**
            *   **Analysis:** High volume of ICMP Echo Requests (Ping) targeting network bandwidth.
            *   **Mitigation:** 
                1. Disable ICMP responses on edge router.
                2. Rate-limit ICMP traffic.
            """,
            "DDoS-UDP_FLOOD": """
            **⚠️ CRITICAL: UDP Volumetric Attack**
            *   **Analysis:** UDP packet storm overwhelming server resources. Likely reflection attack.
            *   **Mitigation:** 
                1. Block UDP fragments.
                2. Implement Geo-blocking if applicable.
            """,
            "SQL_INJECTION": """
            **🚫 ALERT: SQL Injection Attempt**
            *   **Analysis:** Malicious SQL query injected into input field.
            *   **Mitigation:** 
                1. Sanitize all user inputs.
                2. Use prepared statements immediately.
            """
        }
        return reports.get(attack_type, f"**Threat Detected:** {attack_type}\n\nRecommending full packet capture for analysis.")
