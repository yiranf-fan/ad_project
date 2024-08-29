import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from pythresh.thresholds.filter import FILTER as pythresh_FILTER
from pyod.models.combination import maximization
from scipy.stats import mode
from scipy.stats import zscore
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools

from pipeline.ensemble.detectors import *
from pipeline.ensemble.utils import *

class VotingClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom voting classifier that combines the predictions of multiple estimators.

    Parameters:
    estimators (list of tuples): A list of (name, estimator) tuples where name is a string identifier and estimator is a fitted model.
    voting (str, optional): Voting strategy. Options are:
        - 'majority' (default): Majority voting, the class with the most votes wins.
        - 'one_true': At least one true positive results in a positive final prediction.

    Methods:
    fit(X, y=None): Fit all estimators on the data.
    predict(X): Predict class labels for the input data based on the voting strategy.
    """
    def __init__(self, estimators, voting='majority'):
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y=None):
        for name, estimator in self.estimators:
            print(f"Fitting {name} model...")
            estimator.fit(X)
        return self

    def predict(self, X):
        predictions = np.column_stack([estimator.predict(X) for name, estimator in self.estimators])
        
        if self.voting == 'majority':
            final_pred = mode(predictions, axis=1)[0].flatten()
        elif self.voting == 'one_true':
            final_pred = np.max(predictions, axis=1)
        
        return final_pred
    
class ScoreFunctionClassifier(BaseEstimator, ClassifierMixin):
    """
    A custom classifier that aggregates anomaly scores from multiple estimators using various score functions.

    Parameters:
    estimators (list of tuples): A list of (name, estimator) tuples where name is a string identifier and estimator is a fitted model.
    normalization (str, optional): Normalization method to apply to the scores. Options are:
        - 'zscore' (default): Z-score normalization.
        - 'sigmoid': Sigmoid normalization.
    aggregate (str, optional): Aggregation method for combining scores. Options are:
        - 'max' (default): Use the maximum score across estimators.
        - 'sum': Sum the scores across estimators.
    use_filter (bool, optional): Whether to apply an additional filter based on a custom thresholding method. Defaults to False.
    contamination (float, optional): Proportion of outliers in the data. Defaults to 0.1.

    Methods:
    fit(X, y=None): Fit all estimators on the data.
    predict(X): Predict class labels for the input data based on the aggregated scores and filtering.
    """
    def __init__(self, estimators, normalization='zscore', aggregate='max', use_filter = False, contamination=0.1):
        self.estimators = estimators
        self.normalization = normalization
        self.aggregate = aggregate
        self.use_filter = use_filter
        self.contamination = contamination

    def fit(self, X, y=None):
        for name, estimator in self.estimators:
            print(f"Fitting {name} model...")
            estimator.fit(X)
        return self

    def normalize_scores(self, scores):
        if self.normalization == 'zscore':
            return zscore(scores, axis=0)
        elif self.normalization == 'sigmoid':
            return expit(scores)
        else:
            raise ValueError("Unknown normalization method")

    def predict(self, X):
        scores = np.column_stack([estimator.transform(X) for name, estimator in self.estimators])
        
        normalized_scores = self.normalize_scores(scores)
        
        # Aggregate scores using different score function
        if self.aggregate == 'max':
            aggregate_scores = maximization(normalized_scores)
        elif self.aggregate == 'sum':
            aggregate_scores = normalized_scores.sum(axis=1)
        else:
            raise ValueError("Unknown aggregation method")
        
        # Determine the threshold and make predictions
        if self.use_filter:
            thres = pythresh_FILTER()
            final_pred = thres.eval(aggregate_scores)
        else:
            # Determine threshold based on top k% anomalies in training set
            # Assuming 10% contamination with default threshold
            threshold = np.percentile(aggregate_scores, 100 * (1 - self.contamination))
            final_pred = (aggregate_scores > threshold).astype(int)
        
        return final_pred
    
class KMeansMetaLearner(BaseEstimator, ClassifierMixin):
    """
    A meta-learner that uses KMeans clustering for anomaly detection based on distances from cluster centers.

    Parameters:
    n_clusters (int, optional): Number of clusters to form. Defaults to 2.
    percentile_threshold (float, optional): Percentile threshold to classify a point as an anomaly based on its distance from the cluster center. Defaults to 95.
    random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Methods:
    fit(X, y=None): Fit the KMeans model on the data.
    predict(X): Predict class labels for the input data based on the distance to cluster centers.
    """
    def __init__(self, n_clusters=2, percentile_threshold=95, random_state=42):
        self.n_clusters = n_clusters
        self.percentile_threshold = percentile_threshold
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X, y=None):
        print("Fitting KMeans meta learner...")
        self.kmeans.fit(X)
        return self

    def predict(self, X):
        cluster_labels = self.kmeans.predict(X)
        cluster_centers = self.kmeans.cluster_centers_
        
        # Calculate distances from each point to its assigned cluster center
        distances = np.array([np.linalg.norm(x - cluster_centers[cluster]) for x, cluster in zip(X, cluster_labels)])
        
        # Define a threshold for anomaly detection based on the distance percentile
        threshold_distance = np.percentile(distances, self.percentile_threshold)
        
        anomalies = (distances > threshold_distance).astype(int)
        
        return anomalies

def run_pipeline_for_dataset(X_train, 
                             X_test, 
                             y_train, 
                             y_test, 
                             file_path, 
                             estimators_config, 
                             voting='majority', 
                             normalization='zscore', 
                             aggregate='max', 
                             use_filter=True, 
                             contamination=0.1, 
                             verbose=False):
    """
    Run a pipeline for a dataset using multiple estimators with different configurations, including voting and score aggregation.

    Parameters:
    X_train (array-like): Training data feature matrix.
    X_test (array-like): Testing data feature matrix.
    y_train (array-like): Training data labels.
    y_test (array-like): Testing data labels.
    file_path (str): File path for the dataset.
    estimators_config (list): List of estimator configurations.
    voting (str, optional): Voting strategy for VotingClassifier. Defaults to 'majority'.
    normalization (str, optional): Normalization method for ScoreFunctionClassifier. Defaults to 'zscore'.
    aggregate (str, optional): Aggregation method for ScoreFunctionClassifier. Defaults to 'max'.
    use_filter (bool, optional): Whether to apply a custom threshold filter. Defaults to True.
    contamination (float, optional): Proportion of outliers in the data. Defaults to 0.1.
    verbose (bool, optional): Whether to print additional details during execution. Defaults to False.

    Returns:
    metrics (list): List of evaluation metrics for the dataset.
    models (dict): Dictionary of trained models.
    """
    print(f"\nUsing estimator configuration: {estimators_config}")

    detectors = []
    for estimator in estimators_config:
        if isinstance(estimator, dict):
            estimator_name = estimator['name']
            estimator_type = estimator['type']
            estimator_params = estimator.get('params', {})
            
            # Create the detector using detector_factory
            detector = detector_factory(estimator_type, **estimator_params)
            # Store the created detector
            detectors.append((estimator_name, detector))
        else:
            raise ValueError(f"Expected estimator to be a dictionary but got {type(estimator)}")


    voting_clf = VotingClassifier(estimators=detectors, voting=voting)

    score_clf = ScoreFunctionClassifier(estimators=detectors, normalization=normalization, aggregate=aggregate, use_filter=use_filter, contamination=contamination)

    pipeline_voting = Pipeline([('voting_clf', voting_clf)])
    pipeline_score = Pipeline([('score_clf', score_clf)])

    print("Training individual models in pipeline...")
    pipeline_voting.fit(X_train)

    print("Making predictions...")
    final_pred_voting = pipeline_voting.predict(X_test)
    final_pred_score = pipeline_score.predict(X_test)

    predictions = {name: estimator.predict(X_test) for name, estimator in detectors}

    if verbose:
        train_predictions = {name: estimator.fit_predict(X_train) for name, estimator in detectors}
        train_predictions['VotingEnsemble'] = pipeline_voting.predict(X_train)
        train_predictions['ScoreEnsemble'] = pipeline_score.predict(X_train)

        outliers = {name: np.where(pred == 1)[0] for name, pred in predictions.items()}
        outliers['VotingEnsemble'] = np.where(final_pred_voting == 1)[0]
        outliers['ScoreEnsemble'] = np.where(final_pred_score == 1)[0]
        true_outliers = np.where(y_test == 1)[0]

    metrics = []
    models = {}
    for method_name, method_pred in {**predictions, 'VotingEnsemble': final_pred_voting, 'ScoreEnsemble': final_pred_score}.items():
        if verbose:
            train_pred = train_predictions.get(method_name)
            auroc_train = roc_auc_score(y_train, train_pred) if train_pred is not None else None
            avg_precision_train = average_precision_score(y_train, train_pred) if train_pred is not None else None
            f1_train = f1_score(y_train, train_pred) if train_pred is not None else None
            conf_matrix_train = confusion_matrix(y_train, train_pred) if train_pred is not None else None
            method_outliers = outliers.get(method_name, [])

        auroc = roc_auc_score(y_test, method_pred)
        avg_precision = average_precision_score(y_test, method_pred)
        f1 = f1_score(y_test, method_pred)
        conf_matrix = confusion_matrix(y_test, method_pred)
        class_report = classification_report(y_test, method_pred, target_names=['Normal', 'Outlier'])

        if verbose:
            results = {
                'Dataset': file_path,
                'Method': method_name,
                'Estimators Config': estimators_config,
                'Train AUROC': auroc_train,
                'Test AUROC': auroc,
                'Train Averaged Precision': avg_precision_train,
                'Test Averaged Precision': avg_precision,
                'Train F1 Score': f1_train,
                'Test F1 Score': f1,
                'Train Confusion Matrix': conf_matrix_train,
                'Test Confusion Matrix': conf_matrix,
                'Classification Report': class_report,
                'Outliers Indices': method_outliers,
                'True Outliers': true_outliers
            }
        else:
            results = {
                'Dataset': file_path,
                'Method': method_name,
                'Estimators Config': estimators_config,
                'Test AUROC': auroc,
                'Test Averaged Precision': avg_precision,
                'Test F1 Score': f1,
                'Test Confusion Matrix': conf_matrix
            }

        metrics.append(results)

        if method_name == 'VotingEnsemble':
            models[method_name] = pipeline_voting
        elif method_name == 'ScoreEnsemble':
            models[method_name] = pipeline_score

    return metrics, models

def unified_pipeline_train_eval(datasets, 
                                estimators_grid, 
                                voting_options, 
                                normalization_options, 
                                aggregate_options, 
                                test_size=0.2, 
                                random_state=None, 
                                feature_select=False, 
                                n_features=15, 
                                use_vif=True, 
                                vif_threshold=5.0, 
                                upsample=False, 
                                use_autoencoder=False, 
                                code_size=10, 
                                use_filter=True, 
                                contamination=0.1, 
                                verbose=False):
    """
    Run a unified pipeline to train and evaluate models on multiple datasets using various configurations.

    Parameters:
    datasets (list): List of (file_path, label_header) tuples for each dataset.
    estimators_grid (list): List of estimator configurations.
    voting_options (list): List of voting strategies for VotingClassifier.
    normalization_options (list): List of normalization methods for ScoreFunctionClassifier.
    aggregate_options (list): List of aggregation methods for ScoreFunctionClassifier.
    test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
    random_state (int, optional): Random seed for reproducibility. Defaults to None.
    feature_select (bool, optional): Whether to perform feature selection. Defaults to False.
    n_features (int, optional): Number of features to select if feature_select is True. Defaults to 15.
    use_vif (bool, optional): Whether to use VIF for feature selection. Defaults to True.
    vif_threshold (float, optional): VIF threshold for feature selection if use_vif is True. Defaults to 5.0.
    upsample (bool, optional): Whether to upsample the minority class in the training set. Defaults to False.
    use_autoencoder (bool, optional): Whether to use an autoencoder for feature reconstruction. Defaults to False.
    code_size (int, optional): Size of the encoded (latent) layer for the autoencoder. Defaults to 10.
    use_filter (bool, optional): Whether to apply a custom threshold filter. Defaults to True.
    contamination (float, optional): Proportion of outliers in the data. Defaults to 0.1.
    verbose (bool, optional): Whether to print additional details during execution. Defaults to False.

    Returns:
    all_metrics (list): List of evaluation metrics across all datasets and configurations.
    all_models (list): List of trained models across all datasets and configurations.
    """
    all_metrics = []
    all_models = []
    for idx, (file_path, label_header) in enumerate(datasets):
        print(f"\nProcessing Dataset {idx + 1}...")
        X_train, X_test, y_train, y_test = preprocess_and_split_data(file_path, 
                                                                     label_header, 
                                                                     test_size=test_size, 
                                                                     random_state=random_state, 
                                                                     feature_select=feature_select, 
                                                                     n_features=n_features, 
                                                                     use_vif=use_vif, 
                                                                     vif_threshold=vif_threshold, 
                                                                     upsample=upsample, 
                                                                     use_autoencoder=use_autoencoder, 
                                                                     code_size=code_size)
        for voting, normalization, aggregate in itertools.product(voting_options, normalization_options, aggregate_options):
            print(f"\nRunning pipeline with voting={voting}, normalization={normalization}, aggregate={aggregate}")
            for estimators_config in estimators_grid:
                dataset_metrics, dataset_models = run_pipeline_for_dataset(X_train, 
                                                                           X_test, 
                                                                           y_train, 
                                                                           y_test, 
                                                                           file_path, 
                                                                           estimators_config, 
                                                                           voting, 
                                                                           normalization, 
                                                                           aggregate, 
                                                                           use_filter, 
                                                                           contamination, 
                                                                            verbose)
                all_metrics.extend(dataset_metrics)
                for method, model in dataset_models.items():
                    all_models.append((model, max(metric['Test F1 Score'] for metric in dataset_metrics if metric['Method'] == method)))
    
    return all_metrics, all_models

def run_stacking_for_dataset(X_train_main, 
                             X_train_meta, 
                             X_test, 
                             y_train_main, 
                             y_train_meta, 
                             y_test, 
                             file_path, 
                             estimators_config, 
                             n_clusters=2, 
                             percentile_threshold=95, 
                             random_state=42, 
                             verbose=False):
    """
    Run a stacking pipeline for a dataset using a meta-learner (KMeans) on transformed meta-features.

    Parameters:
    X_train_main (array-like): Training data feature matrix for main training.
    X_train_meta (array-like): Training data feature matrix for meta-training.
    X_test (array-like): Testing data feature matrix.
    y_train_main (array-like): Training data labels for main training.
    y_train_meta (array-like): Training data labels for meta-training.
    y_test (array-like): Testing data labels.
    file_path (str): File path for the dataset.
    estimators_config (list): List of estimator configurations.
    n_clusters (int, optional): Number of clusters for the KMeansMetaLearner. Defaults to 2.
    percentile_threshold (float, optional): Percentile threshold for anomaly detection based on distance. Defaults to 95.
    random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    verbose (bool, optional): Whether to print additional details during execution. Defaults to False.

    Returns:
    metrics (list): List of evaluation metrics for the dataset.
    models (dict): Dictionary of trained models.
    """
    print(f"\nUsing estimator configuration: {estimators_config}")
    
    transformed_train_meta = []
    transformed_test_meta = []
    detectors = []
    print("Training individual models in pipeline...")
    for estimator in estimators_config:
        detector = detector_factory(estimator['type'], **estimator['params'])
        detector.fit(X_train_main)
        detectors.append((estimator['name'], detector))
        transformed_train_meta.append(detector.transform(X_train_meta))
        transformed_test_meta.append(detector.transform(X_test))
    
    X_meta_train = np.column_stack(transformed_train_meta)
    X_meta_test = np.column_stack(transformed_test_meta)
    
    meta_learner = KMeansMetaLearner(n_clusters=n_clusters, percentile_threshold=percentile_threshold, random_state=random_state)
    print("Training meta learner...")
    meta_learner.fit(X_meta_train)
    
    print("Making predictions...")
    final_pred = meta_learner.predict(X_meta_test)
    predictions = {name: estimator.predict(X_test) for name, estimator in detectors}

    if verbose:
        train_predictions = {name: estimator.predict(X_train_meta) for name, estimator in detectors}
        train_predictions['StackingEnsemble'] = meta_learner.predict(X_meta_train)
        outliers = {name: np.where(pred == 1)[0] for name, pred in predictions.items()}
        outliers['StackingEnsemble'] = np.where(final_pred == 1)[0]
        true_outliers = np.where(y_test == 1)[0]

    metrics = []
    models = {}
    for method_name, method_pred in {**predictions, 'StackingEnsemble': final_pred}.items():
        if verbose:
            train_pred = train_predictions.get(method_name)
            auroc_train = roc_auc_score(y_train_meta, train_pred) if train_pred is not None else None
            avg_precision_train = average_precision_score(y_train_meta, train_pred) if train_pred is not None else None
            f1_train = f1_score(y_train_meta, train_pred) if train_pred is not None else None
            conf_matrix_train = confusion_matrix(y_train_meta, train_pred) if train_pred is not None else None
            method_outliers = outliers.get(method_name, [])

        auroc = roc_auc_score(y_test, method_pred)
        avg_precision = average_precision_score(y_test, method_pred)
        f1 = f1_score(y_test, method_pred)
        conf_matrix = confusion_matrix(y_test, method_pred)
        class_report = classification_report(y_test, method_pred, target_names=['Normal', 'Outlier'])

        if verbose:
            results = {
                'Dataset': file_path,
                'Method': method_name,
                'Estimators Config': estimators_config,
                'Train AUROC': auroc_train,
                'Test AUROC': auroc,
                'Train Averaged Precision': avg_precision_train,
                'Test Averaged Precision': avg_precision,
                'Train F1 Score': f1_train,
                'Test F1 Score': f1,
                'Train Confusion Matrix': conf_matrix_train,
                'Test Confusion Matrix': conf_matrix,
                'Classification Report': class_report,
                'Outliers Indices': method_outliers,
                'True Outliers': true_outliers
            }
        else:
            results = {
                'Dataset': file_path,
                'Method': method_name,
                'Estimators Config': estimators_config,
                'Test AUROC': auroc,
                'Test Averaged Precision': avg_precision,
                'Test F1 Score': f1,
                'Test Confusion Matrix': conf_matrix
            }

        metrics.append(results)
        
        if method_name == 'StackingEnsemble':
            models[method_name] = meta_learner

    return metrics, models


def pipeline_stacking_train_eval(datasets, 
                                 estimators_grid, 
                                 n_clusters=2, 
                                 percentile_threshold=95, 
                                 test_size=0.2, 
                                 val_size=0.2, 
                                 random_state=None, 
                                 feature_select=False, 
                                 n_features=15, 
                                 use_vif=True, 
                                 vif_threshold=5.0, 
                                 upsample=False, 
                                 use_autoencoder=False, 
                                 code_size=10,
                                 verbose=False):
    """
    Run a stacking pipeline to train and evaluate models on multiple datasets using various configurations.

    Parameters:
    datasets (list): List of (file_path, label_header) tuples for each dataset.
    estimators_grid (list): List of estimator configurations.
    n_clusters (int, optional): Number of clusters for the KMeansMetaLearner. Defaults to 2.
    percentile_threshold (float, optional): Percentile threshold for anomaly detection based on distance. Defaults to 95.
    test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
    val_size (float, optional): Proportion of the training data to include in the validation (meta-training) split. Defaults to 0.2.
    random_state (int, optional): Random seed for reproducibility. Defaults to None.
    feature_select (bool, optional): Whether to perform feature selection. Defaults to False.
    n_features (int, optional): Number of features to select if feature_select is True. Defaults to 15.
    use_vif (bool, optional): Whether to use VIF for feature selection. Defaults to True.
    vif_threshold (float, optional): VIF threshold for feature selection if use_vif is True. Defaults to 5.0.
    upsample (bool, optional): Whether to upsample the minority class in the training set. Defaults to False.
    use_autoencoder (bool, optional): Whether to use an autoencoder for feature reconstruction. Defaults to False.
    code_size (int, optional): Size of the encoded (latent) layer for the autoencoder. Defaults to 10.
    verbose (bool, optional): Whether to print additional details during execution. Defaults to False.

    Returns:
    all_metrics (list): List of evaluation metrics across all datasets and configurations.
    all_models (list): List of trained models across all datasets and configurations.
    """
    all_metrics = []
    all_models = []
    for idx, (file_path, label_header) in enumerate(datasets):
        print(f"\nProcessing Dataset {idx + 1}...")
        X_train_main, X_train_meta, X_test, y_train_main, y_train_meta, y_test = preprocess_and_split_data(file_path, 
                                                                                                           label_header, 
                                                                                                           test_size=test_size, 
                                                                                                           val_size=val_size, 
                                                                                                           random_state=random_state, 
                                                                                                           feature_select=feature_select, 
                                                                                                           n_features=n_features, 
                                                                                                           use_vif=use_vif, 
                                                                                                           vif_threshold=vif_threshold, 
                                                                                                           upsample=upsample, 
                                                                                                           use_autoencoder=use_autoencoder, 
                                                                                                           code_size=code_size)        
        for estimators_config in estimators_grid:
            dataset_metrics, dataset_models = run_stacking_for_dataset(X_train_main, 
                                               X_train_meta, 
                                               X_test, 
                                               y_train_main, 
                                               y_train_meta, 
                                               y_test, 
                                               file_path, 
                                               estimators_config, 
                                               n_clusters=n_clusters, 
                                               percentile_threshold=percentile_threshold, 
                                               random_state=random_state, 
                                               verbose=verbose)
            all_metrics.extend(dataset_metrics)

            for method, model in dataset_models.items():
                all_models.append((model, max(metric['Test F1 Score'] for metric in dataset_metrics if metric['Method'] == method)))
    
    return all_metrics, all_models
