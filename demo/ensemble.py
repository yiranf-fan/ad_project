import streamlit as st
import joblib
import zipfile
from io import BytesIO
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.ensemble.detectors import *
from pipeline.ensemble.pipeline import *
from pipeline.ensemble.utils import *

def ensemble_training_page():
    st.header("Custom Anomaly Detection Model Training")

    # Step 1: Data Upload
    st.subheader("Step 1: Upload Datasets")
    uploaded_files = st.file_uploader("Upload your CSV files", accept_multiple_files=True, type=["csv"])

    if uploaded_files:
        file_buffers = [(file, BytesIO(file.read())) for file in uploaded_files]

        # Preview the data
        for file, buffer in file_buffers:
            buffer.seek(0)
            df = pd.read_csv(buffer)
            st.write(f"**Preview of {file.name}:**")
            st.dataframe(df.head())

    step_2_complete = False
    step_3_complete = False
    step_4_complete = False
    step_5_complete = False
    step_6_complete = False

    # Step 2: Data Selection
    if uploaded_files:
        st.subheader("Step 2: Select Datasets")
        selected_datasets = st.multiselect("Select datasets to use for training", [file.name for file, _ in file_buffers])
        datasets = []
        all_labels_entered = True

        for file, buffer in file_buffers:
            if file.name in selected_datasets:
                label = st.text_input(f"Enter the label column for {file.name}", key=f"label_{file.name}")
                st.markdown(
                    "<p style='font-size: 12px;'>The label is used for evaluation and ranking your models, not during the training process. </p>", 
                    unsafe_allow_html=True
                )
                if label:
                    datasets.append((file, label))
                    st.session_state[file.name] = label
                else:
                    all_labels_entered = False

        if selected_datasets and all_labels_entered:
            step_2_complete = True
    else:
        st.info('Please upload a CSV file to start.')
    
    # Step 3: Framework Selection
    if step_2_complete:
        st.subheader("Step 3: Select Your Framework")
        
        framework_options = ["Please select a framework...", "Independent Ensemble", "Stacking"]
        selected_framework = st.selectbox("Choose the framework for model training", framework_options)

        step_3_complete = selected_framework != "Please select a framework..."

    # Step 4: Preprocessing Options
    if step_3_complete:
        st.subheader("Step 4: Data Preparation Options")
        # Feature Selection
        feature_select = st.checkbox("Enable Feature Selection: Focus on the most important variables")
        n_features = st.number_input(
            "Number of Features to Keep: Retains the top features most relevant to your analysis", 
            min_value=1, 
            max_value=100, 
            value=15, 
            disabled=not feature_select
        )
        
        # VIF
        use_vif = st.checkbox("Use Variance Inflation Factor (VIF): Reduces redundant information among features")
        vif_threshold = st.slider(
            "VIF Threshold: Remove features with multicollinearity above this level", 
            min_value=1.0, 
            max_value=10.0, 
            value=5.0, 
            disabled=not use_vif
        )
        
        # Upsampling
        upsample = st.checkbox("Enable Upsampling: Balance your data by increasing the frequency of underrepresented classes")
        
        # Autoencoder for Feature Reconstruction
        use_autoencoder = st.checkbox("Use Autoencoder: Automatically reduce dimensions while preserving important information")
        code_size = st.number_input(
            "Autoencoder Code Size: Determines the number of dimensions retained by the autoencoder", 
            min_value=1, 
            max_value=100, 
            value=10, 
            disabled=not use_autoencoder
        )
        
        step_4_complete = True
    
    # Step 5: Estimator Grid
    if step_4_complete:
        st.subheader("Step 5: Configure Your Individual Models")

        if "estimators" not in st.session_state:
            st.session_state["estimators"] = []

        display_name_map = {
            'nn': 'Nearest Neighbor',
            'lof': 'Local Outlier Factor',
            'iso': 'Isolation Forest',
            'dbscan': 'DBSCAN'
        }
        reverse_display_name_map = {v: k for k, v in display_name_map.items()}

        # Function to add a new estimator row
        def add_estimator():
            estimator_count = len(st.session_state["estimators"]) + 1
            st.session_state["estimators"].append({
                'name': f'Model {estimator_count}',
                'type': None,
                'params': {}
            })

        # Function to remove an estimator row
        def remove_estimator(index):
            st.session_state["estimators"].pop(index)

        st.button("Add Model", on_click=add_estimator)

        # Display estimators in a table-like format
        for idx, estimator in enumerate(st.session_state["estimators"]):
            st.write(f"**Model {idx + 1}**")
            col1, col2, col3 = st.columns([2, 2, 1])

            # Estimator type with user-friendly names
            with col1:
                display_type = display_name_map.get(estimator['type'], 'Nearest Neighbor')
                selected_display_type = st.selectbox(
                    f"Select Model Type",
                    list(display_name_map.values()),
                    index=list(display_name_map.values()).index(display_type),
                    key=f"type_{idx}"
                )
                estimator['type'] = reverse_display_name_map[selected_display_type]

            # Estimator-specific parameters
            with col2:
                if estimator['type'] == 'nn':
                    estimator['params'] = {'n_neighbors': st.number_input(f"Neighbors (Est. {idx + 1})", value=estimator['params'].get('n_neighbors', 5), key=f"nn_{idx}")}
                elif estimator['type'] == 'lof':
                    estimator['params'] = {
                        'n_neighbors': st.number_input(f"Neighbors (Est. {idx + 1})", value=estimator['params'].get('n_neighbors', 20), key=f"lof_n_{idx}"),
                        'leaf_size': st.number_input(f"Leaf Size (Est. {idx + 1})", value=estimator['params'].get('leaf_size', 30), key=f"lof_l_{idx}"),
                        'contamination': st.number_input(f"Contamination (Est. {idx + 1})", value=estimator['params'].get('contamination', 0.1), key=f"lof_c_{idx}")
                    }
                elif estimator['type'] == 'iso':
                    estimator['params'] = {'contamination': st.number_input(f"Contamination (Est. {idx + 1})", value=estimator['params'].get('contamination', 0.1), key=f"iso_c_{idx}")}
                elif estimator['type'] == 'dbscan':
                    estimator['params'] = {
                        'min_samples': st.number_input(f"Min Samples (Est. {idx + 1})", value=estimator['params'].get('min_samples', 5), key=f"dbscan_m_{idx}"),
                        'metric': st.text_input(f"Metric (Est. {idx + 1})", value=estimator['params'].get('metric', 'euclidean'), key=f"dbscan_metric_{idx}")
                    }

            # Remove button
            with col3:
                st.button("Remove Model", on_click=remove_estimator, args=(idx,), key=f"remove_{idx}")

        estimators_grid = []
        estimators_grid.append(st.session_state["estimators"])
        
        # Proceed to the next steps after all estimators are configured
        step_5_complete = len(st.session_state["estimators"]) > 0

    # Step 6: Framework-Specific Configuration
    if step_5_complete:
        st.subheader("Step 6: Configure Your Framework")

        if selected_framework == "Independent Ensemble":
            voting_map = {
                'One-True Voting (Strict)': 'one_true',
                'Majority Voting (Flexible)': 'majority'
            }
            normalization_map = {
                'Z-Score Normalization (Standard)': 'zscore',
                'Sigmoid Normalization (Compressed)': 'sigmoid'
            }
            aggregate_map = {
                'Max (Pick the highest score)': 'max',
                'Sum (Add up all scores)': 'sum'
            }

            selected_voting_label = st.selectbox("Voting Method: How should the models agree on an anomaly?", list(voting_map.keys()))
            voting_options = voting_map[selected_voting_label]

            selected_normalization_label = st.selectbox("Data Normalization: Adjust data scale for better comparison", list(normalization_map.keys()))
            normalization_options = normalization_map[selected_normalization_label]

            selected_aggregate_label = st.selectbox("Aggregation Method: How to combine model results", list(aggregate_map.keys()))
            aggregate_options = aggregate_map[selected_aggregate_label]

            test_size = st.slider("Test Data Size: Percentage of data used for testing", min_value=0.1, max_value=0.5, value=0.2)
            random_state = st.number_input("Random Seed (Optional): Set a seed for reproducible results", value=None, step=1, format="%d")
            use_filter = st.checkbox("Apply Filter: Use additional filtering on the results", value=True)
            contamination = st.number_input("Expected Anomaly Rate (Contamination): Estimate of anomalies in your data", min_value=0.01, max_value=1.0, value=0.1)

        elif selected_framework == "Stacking":
            n_clusters = st.number_input("Number of Clusters: Group similar data points together", min_value=2, max_value=10, value=2)
            percentile_threshold = st.slider("Percentile Threshold: Sensitivity for detecting anomalies", min_value=90, max_value=99, value=95)
            test_size = st.slider("Test Data Size: Percentage of data used for testing", min_value=0.1, max_value=0.5, value=0.2)
            val_size = st.slider("Validation Data Size: Percentage of data used for validation", min_value=0.1, max_value=0.5, value=0.2)
            random_state = st.number_input("Random Seed (Optional): Set a seed for reproducible results", value=None, step=1, format="%d")

        step_6_complete = True

    # Step 7: Train the Model
    if step_6_complete:
        st.subheader("Step 7: Train Your Customized Model")
        if st.button("Start Training"):
            # Prepare datasets for training by reloading from the buffers
            datasets_to_train = []
            for file, buffer in file_buffers:
                if file.name in selected_datasets:
                    buffer.seek(0)
                    datasets_to_train.append((buffer, st.session_state[file.name]))
            all_metrics = []
            all_models = []

            if selected_framework == "Independent Ensemble":
                st.write("Starting training for independent ensemble framework...")
                metrics, models = unified_pipeline_train_eval(
                    datasets_to_train,
                    estimators_grid, 
                    [voting_options], 
                    [normalization_options], 
                    [aggregate_options],
                    feature_select=feature_select,
                    n_features=n_features, 
                    use_vif=use_vif, 
                    vif_threshold=vif_threshold, 
                    upsample=upsample, 
                    use_autoencoder=use_autoencoder,
                    code_size=code_size, 
                    use_filter=use_filter, 
                    contamination=contamination,
                    test_size=test_size,
                    random_state=random_state,
                    verbose=True
                )
                all_metrics.extend(metrics)
                all_models.extend(models)

            elif selected_framework == "Stacking":
                st.write("Starting training for stacking framework...")
                metrics, models = pipeline_stacking_train_eval(
                    datasets_to_train,
                    estimators_grid, 
                    n_clusters=n_clusters, 
                    percentile_threshold=percentile_threshold, 
                    test_size=test_size, 
                    val_size=val_size, 
                    random_state=random_state, 
                    feature_select=feature_select, 
                    n_features=n_features, 
                    use_vif=use_vif, 
                    vif_threshold=vif_threshold, 
                    upsample=upsample, 
                    use_autoencoder=use_autoencoder, 
                    code_size=code_size,
                    verbose=True
                )
                all_metrics.extend(metrics)
                all_models.extend(models)

            # Display metrics for all selected datasets
            for dataset, label in datasets_to_train:
                file_name = next(file.name for file, _ in file_buffers if buffer == dataset)
                dataset_metrics = [metric for metric in all_metrics if metric['Dataset'] == dataset]
                sorted_metrics = sorted(dataset_metrics, key=lambda x: x['Test F1 Score'], reverse=True)[:3]
                st.write(f"**Top 3 model metrics for {file_name} based on validation F1 score**")
                for i, metric in enumerate(sorted_metrics, 1):
                    print_metrics(i, metric)

            st.write("Training complete!")

            # Save the models and metrics for each dataset
            if not os.path.exists("saved_models"):
                os.makedirs("saved_models")
            
            model_files = []
            for i, (model, score) in enumerate(sorted(all_models, key=lambda x: x[1], reverse=True)[:3], 1):
                model_name = f"top_model_{i}_f1_{score:.3f}.pkl"
                model_path = os.path.join("saved_models", model_name)
                joblib.dump(model, model_path)
                model_files.append(model_path)

            # Save all metrics to a CSV file
            metrics_df = pd.DataFrame(all_metrics)
            metrics_file = "saved_models/training_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            model_files.append(metrics_file)

            # Create a zip file for downloading
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for file in model_files:
                    zip_file.write(file, os.path.basename(file))
            zip_buffer.seek(0)

            # Add a download button for the zip file
            st.download_button(
                label="Download Models and Performance",
                data=zip_buffer,
                file_name="training_metrics_and_models.zip",
                mime="application/zip"
            )
