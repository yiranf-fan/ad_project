import streamlit as st
import pandas as pd
import numpy as np
import time
from io import BytesIO
import plotly.graph_objs as go
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.time_series.detectors import *
from pipeline.time_series.utils import *

def simulate_real_time_processing(df, window_manager, feature_columns, delay=0.1):
    all_data = pd.DataFrame()
    
    plot_placeholder = st.empty()
    
    for time_point, row in df.iterrows():
        point = row[feature_columns].values
        
        # Process the new point and handle missing values
        filled_point = window_manager.process_new_point(point, time_point)
        
        # Use the filled point on the original scale in the dataframe for visualization
        if any(np.isnan(point)):
            row[feature_columns] = filled_point
        
        all_data = pd.concat([all_data, pd.DataFrame([row])])
        anomalies = window_manager.get_anomalies()
        missing_values = window_manager.get_missing_value_points()
        
        with plot_placeholder.container():
            plot_data(all_data, feature_columns, anomalies, missing_values)
        
        time.sleep(delay)
    
    return anomalies, missing_values

def plot_data(df, feature_columns, anomalies, missing_values):
    fig = go.Figure()
    
    for column in feature_columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

    anomaly_x = []
    anomaly_y = []
    missing_x = []
    missing_y = []

    if anomalies:
        for anomaly_time in anomalies:
            anomaly_data = df.loc[anomaly_time]
            anomaly_x.append(anomaly_time)
            anomaly_y.append(anomaly_data[feature_columns].values[0])
    
    if missing_values:
        for missing_time, filled_value in missing_values:
            missing_x.append(missing_time)
            missing_y.append(filled_value[0])

    if anomaly_x:
        fig.add_trace(
            go.Scatter(
                x=anomaly_x,
                y=anomaly_y,
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=5)
            )
        )
    
    if missing_x:
        fig.add_trace(
            go.Scatter(
                x=missing_x,
                y=missing_y,
                mode='markers',
                name='Missing Value',
                marker=dict(color='blue', size=5, symbol='x')
            )
        )
    
    fig.update_layout(
        title='Real-time Time Series Visualization with Anomaly Detection',
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True
    )
    
    st.plotly_chart(fig)

def real_time_anomaly_detection_page():
    st.header('Real-time Time Series Visualization with Anomaly Detection')

    # Section 1: Data Uploading and Selection
    st.subheader('Step 1: Upload Dataset and Select Variables')
    file_path = st.file_uploader('Upload your CSV file', type=['csv'])

    if file_path:
        df_name = file_path.name.split('.')[0]
        
        # Preview the data
        file_buffer = BytesIO(file_path.read())
        df_sample = load_data(file_buffer, header=None)
        st.write(f"**Preview of {df_name}:**")
        st.dataframe(df_sample.head())
        file_buffer.seek(0)

        timestamp_col = st.text_input('Enter the name of the timestamp column (if any)', '')
        
        # Load the data using the helper function
        if timestamp_col:
            df = load_data(file_buffer, timestamp_col=timestamp_col, header='infer')
            df.set_index([timestamp_col], inplace=True)
        else:
            df = load_data(file_buffer, header='infer')

        # Ask the user to select columns
        feature_columns = st.multiselect(
            'Select variables of interests', 
            options=df.columns.tolist()
        )

        st.markdown(
            "<p style='font-size: 12px;'>If you're looking for patterns across multiple variables, select all that apply. If you're focusing on just one specific measurement, choose only that one.</p>", 
            unsafe_allow_html=True
        )

        if feature_columns:
            # Section 2: Data Pre-processing
            st.subheader('Step 2: Data Preparation Options')
            scale_options = {
                'Z-Score Scaling (Standardize data)': 'zscore',
                'Min-Max Scaling (Normalize data)': 'minmax'
            }
            scale_method_label = st.selectbox('Choose a Scaling Method', list(scale_options.keys()))
            scale_method = scale_options[scale_method_label]

            use_pca = st.checkbox('Reduce data dimensions using Principal Component Analysis (PCA)', value=False)

            n_components = st.number_input(
                'Select number of dimensions to keep (only if reducing data)', 
                min_value=1, 
                value=1, 
                format="%d", 
                disabled=not use_pca
            )

            # Section 3: Detector Selection
            st.subheader('Step 3: Detector Selection')
            detector_choice = st.selectbox('Select the anomaly detector you want to use', ('Distance-Based Anomaly Detector', 'Moving Average Anomaly Detector', 'Frequency Domain Anomaly Detector'))

            # Section 4: Detector Configuration
            st.subheader('Step 4: Anomaly Detector Settings')
            st.markdown(
                """
                <style>
                .tooltip {
                    display: inline-block;
                    position: relative;
                    border-bottom: 1px dotted black;
                }

                .tooltip .tooltiptext {
                    visibility: hidden;
                    width: 250px;
                    background-color: #f9f9f9;
                    color: #333;
                    text-align: left;
                    border-radius: 6px;
                    padding: 5px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%; 
                    left: 50%;
                    margin-left: -125px;
                    opacity: 0;
                    transition: opacity 0.3s;
                    font-size: 12px;
                }

                .tooltip:hover .tooltiptext {
                    visibility: visible;
                    opacity: 1;
                }
                </style>

                <div class="tooltip">Anomaly Sensitivity (Threshold) 🛈
                <span class="tooltiptext">This determines how sensitive the detector is to anomalies:
                <br>- A threshold of 2.0 is recommended for Distance-Based and Moving Average Detectors.
                <br>- A threshold of 0.001 is recommended for the Frequency Domain Detector.</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            threshold = st.number_input(
                'Higher values detect fewer anomalies but more significant ones', 
                min_value=0.0, 
                value=2.0, 
                step=0.001, 
                format="%.3f"
            )

            window_size = st.number_input(
                'Data Window Size: How much data is used for each analysis (larger values mean slower but more stable detection)', 
                min_value=10, 
                value=100
            )

            period = st.number_input(
                'Repetition Cycle Length (Period): How often does your data repeat? (Optional)', 
                min_value=0, 
                value=12
            )

            seasonal = st.number_input(
                'Seasonal Adjustment Factor: Adjusts for repeating patterns (Optional)', 
                min_value=7, 
                value=7
            )

            # Ensure seasonal is an odd number and at least 7
            if seasonal < 7:
                st.warning('Seasonal factor must be at least 7. Ajusting to 7.')
                seasonal = 7
            if seasonal % 2 == 0:
                st.warning('Seasonal factor must be an odd number. Adjusting to next higher odd number.')
                seasonal += 1

            robust = st.checkbox(
                'Use Stable Scaling: Helps if if you think your data has many outliers', 
                value=True
            )

            if detector_choice in ['Distance-Based Anomaly Detector', 'Moving Average Anomaly Detector']:
                adaptive_threshold = st.checkbox(
                    'Auto-adjust Detection Sensitivity (Adaptive Threshold): Adjusts sensitivity based on recent data', 
                    value=False
                )

            window_type_options = {
                'Hamming Window': 'hamming',
                'Hann Window': 'hann',
                'Blackman Window': 'blackman',
                'Bartlett Window': 'bartlett'
            }

            if detector_choice == 'Frequency Domain Anomaly Detector':
                window_type_label = st.selectbox('Data Smoothing Window: How to smooth the data before analysis. Hamming window works well with most data.', 
                                                 list(window_type_options.keys()))
                window_type = window_type_options[window_type_label]

                sequence_length = st.number_input(
                    'Sequence Length: How much historical data to use for frequency analysis', 
                    min_value=10, 
                    value=10
                )

            # Initialize the appropriate detector
            detector = None
            if detector_choice == 'Distance-Based Anomaly Detector':
                detector = DistanceDetector(
                    threshold=threshold,
                    window_size=window_size,
                    period=period,
                    seasonal=seasonal,
                    robust=robust,
                    adaptive_threshold=adaptive_threshold,
                    use_pca=use_pca,
                    n_components=n_components if use_pca else None,
                    scale_method=scale_method
                )
            
            elif detector_choice == 'Moving Average Anomaly Detector':
                detector = MADetector(
                    threshold=threshold,
                    window_size=window_size,
                    period=period,
                    seasonal=seasonal,
                    robust=robust,
                    adaptive_threshold=adaptive_threshold,
                    use_pca=use_pca,
                    n_components=n_components if use_pca else None,
                    scale_method=scale_method
                )
            
            elif detector_choice == 'Frequency Domain Anomaly Detector':
                detector = FFTBasedDetector(
                    threshold=threshold,
                    window_size=window_size,
                    period=period,
                    seasonal=seasonal,
                    robust=robust,
                    window_type=window_type,
                    sequence_length=sequence_length,
                    use_pca=use_pca,
                    n_components=n_components if use_pca else None,
                    scale_method=scale_method
                )

            # Add a button to start the detection process
            if st.button('Start Anomaly Detection'):
                if detector:
                    # Initialize MovingWindowManager with the selected detector
                    window_manager = MovingWindowManager(detector)

                    # Simulate real-time processing and display results
                    st.subheader('🚨 Running real-time anomaly detection...')
                    anomalies, missing_values = simulate_real_time_processing(df[feature_columns], window_manager, feature_columns)
                    
                    # Display resulting counts of anomalies and missing values
                    st.write(f"Anomalies detected: {len(anomalies)}")
                    st.write(f"Missing values detected: {len(missing_values)}")

                    # Prepare data for download
                    anomaly_df = pd.DataFrame({
                        'Time': anomalies,
                        'Anomaly': [df.loc[time_point, feature_columns].values if len(feature_columns) > 1 else df.loc[time_point, feature_columns[0]] for time_point in anomalies]
                    })

                    missing_df = pd.DataFrame({
                        'Time': [time for time, _ in missing_values],
                        'Filled Value': [filled[0] if len(filled) == 1 else filled for _, filled in missing_values]
                    })

                    combined_df = pd.concat([anomaly_df, missing_df])
                    file_suffix = feature_columns[0] if len(feature_columns) == 1 else "multi"
                    file_name = f"{df_name}_{file_suffix}_anomalies_missing.csv"

                    # Convert to CSV for download
                    csv = combined_df.to_csv(index=False)

                    # Show download button (even if no anomalies/missing values are detected)
                    st.download_button(label="Download Flagged Anomalies and Missing Values",
                                       data=csv,
                                       file_name=file_name,
                                       mime='text/csv')
                else:
                    st.warning('Please select a detector and configure the options.')
        else:
            st.warning('Please select at least one column to proceed with anomaly detection.')
    
    else:
        st.info('Please upload a CSV file to start.')