import streamlit as st
import pandas as pd
import math
from pathlib import Path
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from real_time import real_time_anomaly_detection_page
from ensemble import ensemble_training_page


st.set_page_config(
    page_title="Generalized Anomaly Detection Toolkit",
    page_icon="👋",
)

st.sidebar.title("Navigation")

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.sidebar.button("Home", key="home"):
    st.session_state.page = "Home"
    st.session_state.selected_tool = None
    st.rerun()

st.session_state.selected_tool = st.sidebar.selectbox(
    "Select a tool:",
    ["Real-time Anomaly Detection", "Custom Anomaly Detection Model Training"],
    index=None,  # Sets the default selection to the first item
    key="tools"
)

if st.session_state.selected_tool:
    st.session_state.page = st.session_state.selected_tool

if st.session_state.page == "Home":
    st.write("# Welcome to Generalized Anomaly Detection Toolkit! 👋")
    st.markdown(
        """
        This toolkit empowers you to perform anomaly detection across various use cases, whether you need real-time detection or custom model training. It leverages a powerful and extendable framework designed to handle the diverse data characteristics encountered in different industries.

        ## What's Inside?

        Our toolkit is built with the flexibility to adapt to specific needs while maintaining robustness across general use cases. Here's what you can expect:
        """
    )

    st.markdown("- **Real-time Anomaly Detection:** Identify and respond to anomalies instantly, ensuring prompt action.")
    with st.expander("Click to view the Real-time Anomaly Detection Architecture"):
        st.image("TSframework.png")

    st.markdown("- **Custom Model Training:** Tailor your detection models with custom thresholds and management tools, fine-tuning them for specific datasets.")
    with st.expander("Click to view the Custom Model Training Architecture"):
        st.image("ADframeworks.png")

    st.markdown(
        """
        ## How to Get Started

        Use the sidebar to navigate through the available tools:

        - **Real-time Anomaly Detection**
        - **Custom Anomaly Detection Model Training**
        """
    )

elif st.session_state.page == "Real-time Anomaly Detection":
    real_time_anomaly_detection_page()
elif st.session_state.page == "Custom Anomaly Detection Model Training":
    ensemble_training_page()