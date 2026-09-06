import streamlit as st

def render(alert_data):
    if not alert_data:
        return
        
    is_alert = alert_data.get("alert", False)
    if is_alert:
        level = alert_data.get("level", "Unknown")
        horizon = alert_data.get("horizon", "Unknown")
        
        st.error(f"🚨 **HAZARDOUS AQI ALERT:** The {horizon} forecast indicates '{level}' air quality. Please take necessary precautions.")
