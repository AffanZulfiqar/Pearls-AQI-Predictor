import streamlit as st
import os
from PIL import Image

def render():
    st.markdown("Feature importance (SHAP) for the trained models:")
    
    artifacts_dir = "artifacts"
    if not os.path.exists(artifacts_dir):
        st.info("SHAP artifacts not found. Run the training pipeline first.")
        return
        
    # Find all SHAP images
    images_found = False
    for item in os.listdir(artifacts_dir):
        if item.startswith("shap_summary_") and item.endswith(".png"):
            images_found = True
            model_type = item.replace("shap_summary_", "").replace(".png", "")
            st.subheader(f"Model: {model_type}")
            image_path = os.path.join(artifacts_dir, item)
            
            try:
                image = Image.open(image_path)
                st.image(image, caption=f"SHAP Feature Importance for {model_type}")
            except Exception as e:
                st.warning(f"Could not load image {item}: {e}")
                
    if not images_found:
        st.info("No SHAP plots found in the artifacts directory.")
