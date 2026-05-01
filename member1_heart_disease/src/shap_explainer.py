import shap
import numpy as np
import matplotlib.pyplot as plt


def explain_model(model, X_background, X_sample, feature_names):
    """
    model: trained keras model
    X_background: small background sample (for SHAP baseline)
    X_sample: data to explain
    feature_names: list of feature names
    """

    # Use DeepExplainer for Keras models
    explainer = shap.DeepExplainer(model, X_background)

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values[0]
    shap.summary_plot(
        shap_values[0],
        X_sample,
        feature_names=feature_names,
        show=True
    )