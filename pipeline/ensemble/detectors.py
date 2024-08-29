import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
import hdbscan
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from pyod.models.thresholds import FILTER as pyod_FILTER


def detector_factory(estimator_type, **params):
    """
    Factory function that creates and returns an anomaly detector based on the given estimator type.

    Parameters:
    estimator_type (str): The type of estimator to create. Options are:
        - 'nn' for NearestNeighborsDetector
        - 'lof' for LocalOutlierFactorDetector
        - 'iso' for IsolationForestDetector
        - 'dbscan' for DBSCANDetector
    **params: Additional keyword arguments specific to the chosen estimator type.

    Returns:
    BaseDetector: An instance of the requested anomaly detector.

    Raises:
    ValueError: If an unknown estimator type is provided.
    """
    if estimator_type == 'nn':
        return NearestNeighborsDetector(**params)
    elif estimator_type == 'lof':
        return LocalOutlierFactorDetector(**params)
    elif estimator_type == 'iso':
        return IsolationForestDetector(**params)
    elif estimator_type == 'dbscan':
        return DBSCANDetector(**params)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")

class BaseDetector(BaseEstimator, TransformerMixin):
    """
    Base class for all custom anomaly detectors. Inherits from scikit-learn's BaseEstimator and TransformerMixin.

    Methods:
    - fit(X, y=None, sample_weight=None): Fits the detector to the data.
    - fit_predict(X, y=None, sample_weight=None): Fits the detector and returns predictions.
    - transform(X): Transforms the data into anomaly scores.
    - predict(X): Predicts whether each sample is an anomaly or not.
    - predict_proba(X): Returns the probability of each sample being an anomaly.
    """
    def fit(self, X, y=None, sample_weight=None):
        self.clf.fit(X, sample_weight=sample_weight)
        return self
    
    def fit_predict(self, X, y=None, sample_weight=None):
        return self.clf.fit_predict(X, sample_weight=sample_weight)
    
    def transform(self, X):
        scores = self.clf.decision_function(X)
        return scores
    
    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        outlier_prob = self.clf.predict_proba(X)
        return outlier_prob
    
class NearestNeighborsDetector(BaseDetector):
    """
    Nearest Neighbors-based anomaly detector. Leverages PyOD's unsupervised nearest neighbors.

    Parameters:
    n_neighbors (int, optional): Number of neighbors to use. Defaults to 5.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.clf = KNN(n_neighbors=n_neighbors, 
                       contamination=pyod_FILTER())

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self
    
    def fit_predict(self, X, y=None):
        return self.clf.labels_
    
    def transform(self, X):
        scores = self.clf.decision_function(X)
        return scores

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        outlier_prob = self.clf.predict_proba(X)
        return outlier_prob

class LocalOutlierFactorDetector(BaseDetector):
    """
    Local Outlier Factor (LOF)-based anomaly detector. Leverages PyOD's LOF.

    Parameters:
    n_neighbors (int, optional): Number of neighbors to use for LOF. Defaults to 20.
    leaf_size (int, optional): Leaf size for tree-based algorithms. Defaults to 30.
    contamination (float, optional): Proportion of outliers in the data. Defaults to 0.1.
    """
    def __init__(self, n_neighbors=20, leaf_size=30, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.contamination = contamination
        self.clf = LOF(n_neighbors=n_neighbors, 
                       leaf_size=leaf_size, 
                       contamination=contamination, 
                       novelty=True)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self
    
    def fit_predict(self, X, y=None):
        return self.clf.labels_

    def transform(self, X):
        scores = self.clf.decision_function(X)
        return scores
    
    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        outlier_prob = self.clf.predict_proba(X)
        return outlier_prob

class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest-based anomaly detector. Leverages PyOD's IForest.

    Parameters:
    contamination (float, optional): Proportion of outliers in the data. Defaults to 0.1.
    """
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.clf = IForest(contamination=contamination)

    def fit(self, X, y=None):
        self.clf.fit(X)
        return self
    
    def fit_predict(self, X, y=None):
        return self.clf.labels_

    def transform(self, X):
        scores = self.clf.decision_function(X)
        return scores

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        outlier_prob = self.clf.predict_proba(X)
        return outlier_prob

class DBSCANDetector(BaseDetector):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise)-based anomaly detector.

    Parameters:
    min_samples (int, optional): Minimum number of samples required in a neighborhood to form a core point. Defaults to 2 * number of features.
    use_eps (bool, optional): Whether to automatically determine the epsilon value using the Knee method. Defaults to False.
    metric (str, optional): Distance metric to use. Defaults to 'euclidean'.
    """
    def __init__(self, min_samples=None, use_eps=False, metric='euclidean'):
        self.min_samples = min_samples
        self.use_eps = use_eps
        self.metric = metric
        self.eps = None
        self.clf = None
    
    def fit(self, X, y=None):
        num_features = X.shape[1]
        
        if self.min_samples is None:
            self.min_samples = num_features * 2

        if self.use_eps:
            n_neighbors = num_features + 1
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric).fit(X)
            distances, indices = nbrs.kneighbors(X)
            distances = np.sort(distances[:, n_neighbors-1], axis=0)
            kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
            self.eps = float(kneedle.elbow_y)
        else:
            self.eps = 0.0

        self.clf = hdbscan.HDBSCAN(min_samples=self.min_samples,
                                   metric=self.metric,
                                   prediction_data=True,
                                   cluster_selection_epsilon=self.eps)
        self.clf.fit(X)
        return self
    
    def fit_predict(self, X, y=None):
        labels = np.where(self.clf.labels_ == -1, 1, 0)
        return labels
    
    def transform(self, X):
        scores = hdbscan.approximate_predict_scores(self.clf, X)
        return scores

    def predict(self, X):
        y_pred, _ = hdbscan.approximate_predict(self.clf, X)
        y_label = np.where(y_pred == -1, 1, 0)
        return y_label
    
    def predict_proba(self, X):
        _, outlier_proba = hdbscan.approximate_predict(self.clf, X)
        return outlier_proba
    
