import json
import logging
import os
import sys

import streamlit as st
import streamlit.components.v1 as components

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings
from src.dashboard.utils import get_dashboard_payload

logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pearls AQI Predictor | Enterprise Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Server-side payload (cached 60 seconds — just the API health check) ───────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_cached_payload():
    return get_dashboard_payload()

payload = fetch_cached_payload()
status  = payload.get("status", {})

# ── Sidebar health/debug panel ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 System Status")
    api_online = status.get("api_online", False)
    hw_conn    = status.get("hopsworks_connected", False)
    fs_avail   = status.get("feature_store_available", False)
    models     = status.get("models_loaded", {})
    data_srcs  = status.get("model_data_sources", {})

    st.markdown(f"**Flask API:** {'🟢 ONLINE' if api_online else '🔴 OFFLINE'}")
    st.markdown(f"**Hopsworks:** {'🟢 CONNECTED' if hw_conn else '🔴 DISCONNECTED'}")
    st.markdown(f"**Feature Store:** {'🟢 AVAILABLE' if fs_avail else '🔴 UNAVAILABLE'}")

    st.markdown("**Models:**")
    for h in ["24h", "48h", "72h"]:
        loaded  = models.get(h, False)
        src     = data_srcs.get(h, "")
        src_tag = f" _{src}_" if src else ""
        icon    = "✅" if loaded else "❌"
        st.markdown(f"  {icon} {h}{src_tag}")

    if status.get("latest_data"):
        st.markdown("**Latest data:**")
        for city, ts in status["latest_data"].items():
            st.markdown(f"  • {city}: `{str(ts)[:16]}`")

    if not api_online:
        st.error(
            f"Flask API is not reachable at:\n`{payload['flask_api_url']}`\n\n"
            "Start Flask with:\n```\npython -m src.inference.api\n```"
        )

    st.divider()
    st.caption(f"Server time: {payload.get('generated_at', '')}")
    st.caption(f"API URL: `{payload.get('flask_api_url', '')}`")

# ── Main dashboard (fullscreen HTML bundle) ───────────────────────────────────

# Inject CSS to remove Streamlit chrome
st.markdown("""
<style>
    header[data-testid="stHeader"] { display: none !important; }
    footer                         { display: none !important; }
    #MainMenu                      { visibility: hidden !important; }
    .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100vw !important;
        overflow: hidden !important;
    }
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

web_dir    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../web"))
index_path = os.path.join(web_dir, "index.html")
style_path = os.path.join(web_dir, "style.css")
js_path    = os.path.join(web_dir, "app.js")

try:
    with open(style_path, "r", encoding="utf-8") as f:
        css = f.read()
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Inline CSS + inject server payload + inline JS
    bundled = html.replace(
        '<link rel="stylesheet" href="style.css">',
        f"<style>{css}</style>"
    )
    server_data_json = json.dumps(payload)
    bundled = bundled.replace(
        '<script src="app.js"></script>',
        f"<script>window.__SERVER_DATA__ = {server_data_json};</script>"
        f"<script>{js}</script>"
    )

    components.html(bundled, height=730, scrolling=False)

except FileNotFoundError as e:
    st.error(f"Could not load dashboard template: {e}")
except Exception as e:
    st.error(f"Dashboard error: {e}")
    logger.exception("Dashboard rendering failed")
