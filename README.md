# Anomaly Detection Tool

This repository provides a robust foundation for detecting anomalies in various time series datasets using a moving window approach and offers powerful ensemble learning techniques for enhancing anomaly detection across diverse datasets.

## Part 1: Time-Series Anomaly Detection

### Data Pre-processing

1. **Standardization**  
   Scales the data using either z-score or min-max scaling, with optional PCA for dimensionality reduction.

2. **Seasonal-Trend Decomposition**  
   Decomposes time series data into seasonal, trend, and residual components. The residuals are used as input for distance detector and moving average approaches.

### Sliding Window Framework

1. **New Data Processing**  
   Processes data points one by one, updating the model and checking for anomalies in real-time.

2. **Missing Value Handling**  
   Fills missing values using the last valid data point within the window and tracks these points for reporting.

### Anomaly Detectors

The detectors are encapsulated in a common interface:

1. **Distance-based Detector**
   - Detects anomalies using Mahalanobis distance (multivariate) or z-score (univariate).
   - **Feature:** Optional adaptive thresholding adjusts dynamically based on residual statistics.

2. **Moving Average-based Detector**
   - Compares the current moving average with historical averages to detect significant deviations.
   - **Feature:** Optional adaptive thresholding.

3. **Frequency-based Detector**
   - Analyzes frequency patterns using Fast Fourier Transform (FFT) and cross-correlation.
   - Detects anomalies by measuring the similarity between current and previous frequency spectra.

## Part 2: Ensemble Training for Anomaly Detection

### Data Pre-processing

1. **Feature Selection & VIF Filtering**
   - Identifies and retains the most relevant features, ensuring robust model performance.

2. **Up-sampling & Feature Reconstruction**
   - Uses SMOTE for up-sampling and Autoencoders for reconstructing features.

3. **Mahalanobis Distance**
   - Adds Mahalanobis distance as a feature to enhance detection capabilities.

### Ensemble Frameworks

1. **Voting Ensemble (Independent Ensemble)**
   - Aggregates predictions using majority voting or one true voting for final anomaly detection.

2. **Score Function Ensemble (Independent Ensemble)**
   - Aggregates (max or sum score functions) and normalizes (z-score or sigmoid scaling) anomaly scores, then applies thresholding for detection.

3. **Stacking Ensemble**
   - Combines base detector predictions using a meta-learner, such as K-Means, to detect anomalies based on clustering.

### Supported Detectors

These detectors, integrated through a common interface, include:

1. **Nearest Neighbors Detector (NN)**
   - Detects anomalies by measuring the distance to the nearest neighbors. Points with a large distance from their neighbors are flagged as anomalies.

2. **Local Outlier Factor Detector (LOF)**
   - Identifies anomalies by comparing the local density of a point to the densities of its neighbors. Points with significantly lower density than their neighbors are flagged as anomalies.

3. **Isolation Forest Detector (ISO)**
   - Constructs a series of decision trees by randomly selecting features and splitting values. Anomalies are isolated quickly due to their distinct attributes, resulting in shorter paths within the trees.

4. **DBSCAN Detector**
   - Groups points into dense clusters and identifies points that do not belong to any cluster as anomalies.


## Running on Your Own Machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Ensure that the datasets and configurations are set correctly according to the instructions provided in the documentation.

3. Customize the parameters for the detectors, data preprocessing, and ensemble methods as needed.

4. Run the pipeline using the command:

    ```
   $ streamlit run demo/streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false
   ```