import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_shap_artifacts(model, X_train, X_test, feature_names, model_type, output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    
    # SHAP can be tricky depending on the pipeline/wrapper.
    try:
        if model_type == "random_forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            X_eval = X_test
        elif model_type == "ridge":
            # Ridge is inside a pipeline
            ridge_model = model.named_steps['ridge']
            scaler = model.named_steps['scaler']
            explainer = shap.LinearExplainer(ridge_model, scaler.transform(X_train))
            shap_values = explainer.shap_values(scaler.transform(X_test))
            X_eval = X_test
        else:
            # For tf_model wrapper, use KernelExplainer on a summary of background
            background = shap.sample(X_train, 100)
            explainer = shap.KernelExplainer(model.predict, background)
            X_eval = X_test[:50] # Limit to speed up kernel explainer
            shap_values = explainer.shap_values(X_eval)
            
        plt.figure()
        shap.summary_plot(shap_values, X_eval, feature_names=feature_names, show=False)
        
        plot_path = os.path.join(output_dir, f"shap_summary_{model_type}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        
        return plot_path
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to generate SHAP artifact for {model_type}: {e}")
        return None
