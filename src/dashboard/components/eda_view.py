import streamlit as st
import hopsworks
from src.config import settings
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

@st.cache_data(ttl=3600)
def fetch_history(city_id):
    try:
        project = hopsworks.login(
            project=settings.HOPSWORKS_PROJECT_NAME,
            api_key_value=settings.HOPSWORKS_API_KEY
        )
        fs = project.get_feature_store()
        fg = fs.get_feature_group(settings.FEATURE_GROUP_NAME, settings.FEATURE_GROUP_VERSION)
        
        df = fg.select_all().read()
        df = df[df["city_id"] == city_id].sort_values("timestamp")
        return df.tail(24 * 7) # last 7 days
    except Exception as e:
        return pd.DataFrame()

def render(city_id):
    df = fetch_history(city_id)
    if df.empty:
        st.info("Historical data not available for this city. Run the backfill pipeline first.")
        st.code("python -m src.backfill.run_backfill --days 60", language="bash")
        return
    
    st.markdown("#### 📊 Exploratory Data Analysis")
    
    # Chart 1: AQI Trend Over Time (last 7 days)
    fig1 = px.line(
        df, x="timestamp", y="aqi",
        title="AQI Trend (Last 7 Days)",
        labels={"timestamp": "Time", "aqi": "AQI"}
    )
    fig1.add_hline(
        y=settings.HAZARDOUS_AQI_THRESHOLD,
        line_dash="dash", line_color="red",
        annotation_text="Unhealthy Threshold"
    )
    fig1.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Chart 2: Average AQI by Hour of Day
    with col1:
        if "hour" in df.columns:
            hourly_avg = df.groupby("hour")["aqi"].mean().reset_index()
            fig2 = px.bar(
                hourly_avg, x="hour", y="aqi",
                title="Average AQI by Hour of Day",
                labels={"hour": "Hour (0-23)", "aqi": "Mean AQI"},
                color="aqi",
                color_continuous_scale="YlOrRd"
            )
            fig2.update_layout(template="plotly_dark", height=350, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
    
    # Chart 3: Pollutant Correlation Heatmap
    with col2:
        pollutant_cols = [c for c in ["aqi", "pm2_5", "pm10", "no2", "so2", "co", "o3", 
                                       "temperature", "humidity", "wind_speed"] if c in df.columns]
        if len(pollutant_cols) >= 3:
            corr = df[pollutant_cols].corr()
            fig3 = px.imshow(
                corr,
                text_auto=".2f",
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig3.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig3, use_container_width=True)
    
    # Chart 4: AQI Distribution
    fig4 = px.histogram(
        df, x="aqi", nbins=30,
        title="AQI Distribution",
        labels={"aqi": "AQI Value", "count": "Frequency"},
        color_discrete_sequence=["#3a7bd5"]
    )
    fig4.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig4, use_container_width=True)
