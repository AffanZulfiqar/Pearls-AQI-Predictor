import streamlit as st

def get_color(category):
    colors = {
        "Good": "#00e400",
        "Moderate": "#ffff00",
        "Unhealthy for Sensitive Groups": "#ff7e00",
        "Unhealthy": "#ff0000",
        "Very Unhealthy": "#8f3f97",
        "Hazardous": "#7e0023"
    }
    return colors.get(category, "#808080")

def render(forecasts):
    if not forecasts:
        st.warning("No forecast data available.")
        return
        
    cols = st.columns(len(forecasts))
    
    for i, (horizon, data) in enumerate(forecasts.items()):
        with cols[i]:
            val = data["value"]
            cat = data["category"]
            color = get_color(cat)
            
            st.markdown(
                f"""
                <div style="background-color: {color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {color}; margin-bottom: 10px;">
                    <h3 style="margin-top:0;">{horizon} Forecast</h3>
                    <p style="font-size: 36px; font-weight: bold; margin: 10px 0; color: {color}">{val}</p>
                    <p style="font-size: 18px; margin: 0;">{cat}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
