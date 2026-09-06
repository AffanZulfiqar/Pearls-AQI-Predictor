import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings
from src.dashboard.utils import get_dashboard_payload

# Import modular components per project specification
try:
    from src.dashboard.components import forecast_view, eda_view, shap_view, alert_banner
except ImportError:
    forecast_view = None
    eda_view = None
    shap_view = None
    alert_banner = None

# Page configuration
st.set_page_config(
    page_title="Pearls AQI Predictor | Enterprise Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fetch server-side dashboard telemetry (cached for 10 minutes)
@st.cache_data(ttl=600, show_spinner=False)
def fetch_cached_payload():
    return get_dashboard_payload()

payload = fetch_cached_payload()

# Check if user requested modular components inspection view
query_params = st.query_params
if query_params.get("view") == "modular":
    st.title("💎 Pearls AQI Predictor — Modular Streamlit View")
    st.caption(f"Status: {payload['source']} • Generated: {payload.get('generated_at', '')}")
    
    col_nav1, col_nav2 = st.columns([3, 1])
    with col_nav1:
        city_options = [c["id"] for c in settings.CITIES]
        selected_city = st.selectbox("Select City", options=city_options, format_func=lambda x: x.capitalize())
    with col_nav2:
        st.write("")
        st.write("")
        st.markdown("[Switch to Executive Glassmorphism UI](?)")
    
    city_data = payload["cities"].get(selected_city, payload["cities"]["islamabad"])
    
    # Render modular alerts
    if alert_banner:
        is_hazard = city_data["aqi"] > settings.HAZARDOUS_AQI_THRESHOLD
        alert_banner.render({
            "alert": is_hazard,
            "level": city_data["status"],
            "horizon": "Current"
        })
        
    # Render modular 3-day forecast
    if forecast_view:
        st.subheader("3-Day Atmospheric Forecast")
        fc_dict = {}
        for h, text in city_data["forecasts"].items():
            # Parse 'AQI 58 - Good'
            parts = text.replace("AQI ", "").split(" - ")
            val = int(parts[0]) if parts[0].isdigit() else 60
            cat = parts[1] if len(parts) > 1 else "Moderate"
            fc_dict[h] = {"value": val, "category": cat}
        forecast_view.render(fc_dict)
        
    # Render modular SHAP feature importance
    if shap_view:
        st.subheader("SHAP Feature Importance")
        shap_view.render()
        
    # Render modular EDA charts from Feature Store
    if eda_view:
        st.subheader("Historical Telemetry & EDA")
        eda_view.render(selected_city)

else:
    # Full-screen CSS override - adapt to any display resolution without vertical scrolling
    st.markdown("""
    <style>
        /* Completely hide Streamlit UI bars and gutters */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        footer {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
        }
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100vw !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }
        iframe {
            width: 100vw !important;
            height: 100vh !important;
            min-height: 100vh !important;
            max-height: 100vh !important;
            border: none !important;
            overflow: hidden !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Path to the standalone web assets
    web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../web"))
    index_path = os.path.join(web_dir, "index.html")
    style_path = os.path.join(web_dir, "style.css")
    app_js_path = os.path.join(web_dir, "app.js")

    try:
        with open(style_path, "r", encoding="utf-8") as f:
            css_content = f.read()

        with open(app_js_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        with open(index_path, "r", encoding="utf-8") as f:
            html_template = f.read()

        # Serialize server-side live data
        json_data = json.dumps(payload)

        # Inline CSS and JS into single standalone HTML payload with injected server data
        bundled_html = html_template.replace('<link rel="stylesheet" href="style.css">', f'<style>{css_content}</style>')
        bundled_html = bundled_html.replace(
            '<script src="app.js"></script>',
            f'<script>window.__SERVER_DATA__ = {json_data};</script><script>{js_content}</script>'
        )

        # Render inside Streamlit fullscreen iframe with responsive height
        components.html(bundled_html, height=730, scrolling=False)

    except Exception as e:
        st.error(f"Failed to load Pearls AQI Dashboard template: {e}")
