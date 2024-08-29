import pandas as pd
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.seasonal import STL
from scipy.fftpack import fft
from scipy.signal import get_window, correlate
from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors that use a sliding window approach for real-time detection.

    Parameters:
    threshold (float): Threshold value for anomaly detection.
    window_size (int): Size of the sliding window.
    period (int, optional): Period for seasonal decomposition. If None, it will be estimated. Defaults to None.
    seasonal (int, optional): Seasonal component for STL decomposition. Defaults to 7.
    robust (bool, optional): Whether to use robust STL decomposition. Defaults to True.
    adaptive_threshold (bool, optional): Whether to use an adaptive threshold based on the residuals. Defaults to False.
    use_pca (bool, optional): Whether to apply PCA to the standardized data. Defaults to False.
    n_components (int, optional): Number of PCA components to retain if use_pca is True. Defaults to None.
    scale_method (str, optional): Scaling method for data. Options are 'zscore' or 'minmax'. Defaults to 'zscore'.
    """
    def __init__(self, threshold, window_size, period=None, seasonal=7, robust=True, adaptive_threshold=False, use_pca=False, n_components=None, scale_method='zscore'):
        self.threshold = threshold
        self.window_size = window_size
        self.period = period
        self.seasonal = seasonal
        self.robust = robust
        self.adaptive_threshold = adaptive_threshold
        self.raw_data_window = deque(maxlen=window_size)
        self.residuals_window = deque(maxlen=window_size)
        self.anomaly_scores = []
        self.mean_distr = None
        self.cov_matrix_inv = None
        self.old_mean_distr = None
        self.scaler = None
        self.pca = None
        self.use_pca = use_pca
        self.n_components = n_components
        self.scale_method = scale_method

    def standardize_initial_window(self, initial_data):
        """
        Standardize the initial data window and optionally apply PCA.

        Parameters:
        initial_data (np.array): Initial data to standardize.

        Returns:
        np.array: Standardized (and optionally PCA-transformed) data.
        """
        if self.scale_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scale_method == 'zscore':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling type")
        
        standardized_data = self.scaler.fit_transform(initial_data)
        
        if self.use_pca:
            self.pca = PCA(n_components=self.n_components)
            standardized_data = self.pca.fit_transform(standardized_data)
        
        return standardized_data

    def standardize_new_point(self, new_point):
        """
        Standardize a new data point based on the previously fitted scaler and PCA (if applicable).

        Parameters:
        new_point (np.array): New data point to standardize.

        Returns:
        np.array: Standardized (and optionally PCA-transformed) data point.
        """
        if self.scaler is not None:
            transformed_point = self.scaler.transform([new_point])
            if self.pca is not None:
                transformed_point = self.pca.transform(transformed_point)
            return transformed_point[0]
        else:
            raise ValueError("Scaler has not been initialized. Ensure the initial window is standardized first.")

    @abstractmethod
    def detect(self, new_point):
        pass
    
    def handle_missing_values(self, new_point, time_point):
        """
        Handle missing values in the new data point by forward filling.

        Parameters:
        new_point (np.array): New data point to check for missing values.
        time_point (str or pd.Timestamp): Time point of the data.

        Returns:
        np.array: Data point with missing values handled.
        """
        if any(np.isnan(new_point)):
            print(f"Missing value detected at {time_point}. Applying forward fill.")
            if len(self.raw_data_window) > 0:
                last_valid = np.array(self.raw_data_window)[-1]
                new_point = np.where(np.isnan(new_point), last_valid, new_point)
            else:
                raise ValueError("Missing values detected with no prior data to fill.")
        return new_point

    def stl_decompose(self, ts):
        """
        Perform STL decomposition on the time series data to extract residuals.

        Parameters:
        ts (pd.Series): Time series data.

        Returns:
        np.array: Residuals from the STL decomposition.
        """
        if self.period is None:
            freqs = np.fft.fftfreq(len(ts))
            fft_values = np.fft.fft(ts.values)
            period = abs(freqs[np.argmax(np.abs(fft_values))])
            period = int(round(1 / period))
            self.period = period
            print(f"Estimated period: {period}")
        
        stl = STL(ts, period=self.period, seasonal=self.seasonal, robust=self.robust)
        result = stl.fit()
        residuals = result.resid

        return residuals

    def update_residuals_window(self):
        """
        Update the residuals window with decomposed residuals from the raw data window.
        """
        data_array = np.array(self.raw_data_window)
        decomposed_residuals = []
        for i in range(data_array.shape[1]):
            residuals = self.stl_decompose(pd.Series(data_array[:, i]))
            decomposed_residuals.append(residuals)
        residuals_matrix = np.array(decomposed_residuals).T
        self.residuals_window.clear()
        self.residuals_window.extend(residuals_matrix)

    def calculate_adaptive_threshold(self, residuals):
        """
        Calculate an adaptive threshold based on the residuals.

        Parameters:
        residuals (np.array): Residuals to calculate the threshold.

        Returns:
        float: Calculated adaptive threshold.
        """
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)
        return mean_resid + 2 * std_resid

    def smooth_anomaly_scores(self, scores, window_size=5):
        """
        Smooth the anomaly scores using a moving average.

        Parameters:
        scores (list): List of anomaly scores.
        window_size (int, optional): Window size for smoothing. Defaults to 5.

        Returns:
        np.array: Smoothed anomaly scores.
        """
        return np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

    def update_distribution(self):
        """
        Update the distribution (mean and inverse covariance) based on the current residuals window.
        """
        data = np.array(self.residuals_window)
        try:
            lw = LedoitWolf()
            lw.fit(data)
            reg_cov_matrix = lw.covariance_
            self.cov_matrix_inv = np.linalg.inv(reg_cov_matrix)
            self.old_mean_distr = self.mean_distr if self.mean_distr is not None else data.mean(axis=0)
            self.mean_distr = data.mean(axis=0)
        except np.linalg.LinAlgError:
            reg_cov_matrix = lw.covariance_ + np.eye(lw.covariance_.shape[0]) * 1e-5
            self.cov_matrix_inv = np.linalg.inv(reg_cov_matrix)
            self.old_mean_distr = self.mean_distr if self.mean_distr is not None else data.mean(axis=0)
            self.mean_distr = data.mean(axis=0)

class DistanceDetector(BaseDetector):
    """
    Anomaly detector that uses Mahalanobis distance or z-score for point outlier detection.

    Parameters:
    Inherits parameters from BaseDetector.
    """
    def __init__(self, threshold, window_size, period=None, seasonal=7, robust=True, adaptive_threshold=False, use_pca=False, n_components=None, scale_method='zscore'):
        super().__init__(threshold, window_size, period, seasonal, robust, adaptive_threshold, use_pca, n_components, scale_method)
        self.univariate = None

    def fit(self, data):
        """
        Fit the detector with initial data.

        Parameters:
        data (np.array): Initial data to fit the detector.

        Raises:
        ValueError: If the initial data size is less than the window size.
        """
        if len(data) < self.window_size:
            raise ValueError("Initial data size must be at least as large as the window size.")
        self.univariate = data.shape[1] == 1
        standardized_data = self.standardize_initial_window(data[:self.window_size])
        self.raw_data_window.extend(standardized_data)
        self.update_residuals_window()
        if not self.univariate:
            self.update_distribution()

    def calculate_anomaly_score(self, new_point):
        """
        Calculate the anomaly score for a new data point.

        Parameters:
        new_point (np.array): New data point to evaluate.

        Returns:
        float: Anomaly score for the new data point.
        """
        standardized_point = self.standardize_new_point(new_point)
        self.raw_data_window.append(standardized_point)
        self.update_residuals_window()

        residuals = np.array(self.residuals_window).flatten() if self.univariate else np.array(self.residuals_window)
        
        if self.univariate:
            current_z_score = (residuals[-1] - np.mean(residuals)) / np.std(residuals)
            return abs(current_z_score)
        else:
            self.update_distribution()
            if self.old_mean_distr is None:
                return 0
            
            new_point_residuals = np.array([residuals[-1] for residuals in residuals.T])
            old_distance = mahalanobis(new_point_residuals, self.old_mean_distr, self.cov_matrix_inv)
            new_distance = mahalanobis(new_point_residuals, self.mean_distr, self.cov_matrix_inv)
            return abs(new_distance - old_distance)

    def detect(self, new_point):
        """
        Detect if a new data point is an anomaly based on the calculated anomaly score.

        Parameters:
        new_point (np.array): New data point to evaluate.

        Returns:
        bool: Whether the new point is considered an anomaly.
        """
        if len(self.raw_data_window) < self.window_size:
            self.raw_data_window.append(self.standardize_new_point(new_point))
            return False

        score = self.calculate_anomaly_score(new_point)
        self.anomaly_scores.append(score)
        smoothed_scores = self.smooth_anomaly_scores(self.anomaly_scores)
        
        is_anomaly = False
        if self.adaptive_threshold:
            adaptive_threshold = self.calculate_adaptive_threshold(smoothed_scores)
            print(f'Adpative_threshold: {adaptive_threshold}')
            is_anomaly = smoothed_scores[-1] > adaptive_threshold
        else:
            is_anomaly = smoothed_scores[-1] > self.threshold

        if not is_anomaly:
            self.update_distribution()

        return is_anomaly
        
class MADetector(BaseDetector):
    """
    Anomaly detector that uses a moving average comparison for point outlier detection.

    Parameters:
    Inherits parameters from BaseDetector.
    """
    def __init__(self, threshold, window_size, period=None, seasonal=7, robust=True, adaptive_threshold=False, use_pca=False, n_components=None, scale_method='zscore'):
        super().__init__(threshold, window_size, period, seasonal, robust, adaptive_threshold, use_pca, n_components, scale_method)
        self.current_ma = None

    def fit(self, data):
        """
        Fit the detector with initial data.

        Parameters:
        data (np.array): Initial data to fit the detector.

        Raises:
        ValueError: If the initial data size is less than the window size.
        """
        if len(data) < self.window_size:
            raise ValueError("Initial data size must be at least as large as the window size.")
        standardized_data = self.standardize_initial_window(data[:self.window_size])
        self.raw_data_window.extend(standardized_data)
        self.update_residuals_window()
        self.current_ma = np.mean(np.array(self.residuals_window), axis=0)

    def detect(self, new_point):
        """
        Detect if a new data point is an anomaly based on the moving average comparison.

        Parameters:
        new_point (np.array): New data point to evaluate.

        Returns:
        bool: Whether the new point is considered an anomaly.
        """
        standardized_point = self.standardize_new_point(new_point)
        if len(self.raw_data_window) < self.window_size:
            self.raw_data_window.append(standardized_point)
            return False
    
        if self.current_ma is None:
            raise ValueError("The detector has not been fitted with initial data.")
        
        old_ma = self.current_ma.copy()
        self.raw_data_window.append(standardized_point)
        self.update_residuals_window()
        self.current_ma = np.mean(np.array(self.residuals_window), axis=0)
        
        ma_change = np.linalg.norm(self.current_ma - old_ma)
        self.anomaly_scores.append(ma_change)
        smoothed_scores = self.smooth_anomaly_scores(self.anomaly_scores)
        
        is_anomaly = False
        if self.adaptive_threshold:
            adaptive_threshold = self.calculate_adaptive_threshold(smoothed_scores)
            print(f'Adaptive_threshold: {adaptive_threshold}')
            is_anomaly = smoothed_scores[-1] > adaptive_threshold
        else:
            is_anomaly = smoothed_scores[-1] > self.threshold
        
        if not is_anomaly:
            self.current_ma = np.mean(np.array(self.residuals_window), axis=0)
        
        return is_anomaly

class FFTBasedDetector(BaseDetector):
    """
    Anomaly detector that uses FFT and cross-correlation for sequence outlier detection.

    Parameters:
    Inherits parameters from BaseDetector, with additional parameters for FFT-based detection.
    window_type (str, optional): Type of window function to apply. Defaults to 'hamming'.
    sequence_length (int, optional): Length of the sequence for FFT. Defaults to 10.
    """
    def __init__(self, threshold, window_size, period=None, seasonal=7, robust=True, window_type='hamming', sequence_length=10, use_pca=False, n_components=None, scale_method='zscore'):
        super().__init__(threshold, window_size, period, seasonal, robust, use_pca=use_pca, n_components=n_components, scale_method=scale_method)
        self.window_type = window_type
        self.sequence_length = sequence_length
        self.sequence_buffer = deque(maxlen=sequence_length)
        self.window_function = get_window(self.window_type, sequence_length)
        self.prev_yf = None
        
    def calculate_fft(self, window_data):
        """
        Calculate the FFT of the windowed data.

        Parameters:
        window_data (np.array): Windowed data to compute FFT.

        Returns:
        np.array: FFT of the windowed data.
        """
        N = len(window_data)
        combined_signal = np.sum(window_data, axis=1)
        windowed_signal = combined_signal * self.window_function
        yf = fft(windowed_signal)
        return yf

    def fit(self, data):
        """
        Fit the detector with initial data.

        Parameters:
        data (np.array): Initial data to fit the detector.
        """
        standardized_data = self.standardize_initial_window(data[:self.window_size])
        self.raw_data_window.extend(standardized_data)
        self.sequence_buffer.extend(standardized_data[-self.sequence_length:])
        self.prev_yf = self.calculate_fft(np.array(list(self.raw_data_window)[-self.sequence_length:]))
        print(f"Initial sequence buffer filled with {len(self.sequence_buffer)} points.")

    def detect(self, new_point):
        """
        Detect if a new data point is an anomaly based on FFT and cross-correlation.

        Parameters:
        new_point (np.array): New data point to evaluate.

        Returns:
        bool: Whether the new point is considered an anomaly.
        """
        standardized_point = self.standardize_new_point(new_point)
        self.sequence_buffer.append(standardized_point)
        
        if len(self.sequence_buffer) < self.sequence_length:
            print(f"Sequence buffer not yet full: {len(self.sequence_buffer)}/{self.sequence_length} points.")
            return False
        
        current_sequence = np.array(self.sequence_buffer)

        if len(self.raw_data_window) < self.sequence_length:
            print(f"Not enough data in the raw_data_window to form a previous sequence.")
            return False

        previous_sequence = np.array(list(self.raw_data_window)[-self.sequence_length:])
        self.raw_data_window.extend(current_sequence)
        if len(self.raw_data_window) > self.window_size:
            self.raw_data_window = deque(list(self.raw_data_window)[-self.window_size:], maxlen=self.window_size)
        
        if self.prev_yf is None or not np.array_equal(previous_sequence, list(self.raw_data_window)[-self.sequence_length-1:-1]):
            self.prev_yf = self.calculate_fft(previous_sequence)

        curr_yf = self.calculate_fft(current_sequence)

        prev_magnitude = np.abs(self.prev_yf)
        curr_magnitude = np.abs(curr_yf)

        band_start, band_end = 0, 50
        prev_band = prev_magnitude[band_start:band_end]
        curr_band = curr_magnitude[band_start:band_end]

        cross_corr = correlate(curr_band, prev_band, mode='valid')
        cross_corr_value = cross_corr[0] / (np.linalg.norm(prev_band) * np.linalg.norm(curr_band))

        deviation = np.abs(1 - cross_corr_value)
        
        if deviation > self.threshold:
            print(f"Cross-correlation value: {cross_corr_value:.4f} | Deviation: {deviation:.4f}")
            return True
        return False

class MovingWindowManager:
    """
    Manager for applying detectors to time series data in a moving window manner.

    Parameters:
    detector (BaseDetector): Anomaly detector to manage.
    """
    def __init__(self, detector):
        self.detector = detector
        self.anomalies = []
        self.missing_value_points = []

    def process_new_point(self, point, time_point):
        """
        Process a new data point for anomaly detection and missing value handling.

        Parameters:
        point (np.array): New data point to process.
        time_point (str or pd.Timestamp): Time point associated with the data.

        Returns:
        np.array: Processed data point with missing values handled.
        """
        original_point = point.copy()

        if any(np.isnan(point)):
            filled_point = self.detector.handle_missing_values(original_point, time_point)
            
            if len(self.detector.raw_data_window) < self.detector.window_size:
                self.missing_value_points.append((time_point, filled_point.copy()))
            else:
                if self.detector.scaler is not None:
                    inverse_scaled_point = self.detector.scaler.inverse_transform([filled_point])[0]
                    self.missing_value_points.append((time_point, inverse_scaled_point.copy()))
                else:
                    self.missing_value_points.append((time_point, filled_point.copy()))
            
            point = filled_point
        else:
            filled_point = original_point

        if len(self.detector.raw_data_window) < self.detector.window_size:
            self.detector.raw_data_window.append(point)
            if len(self.detector.raw_data_window) == self.detector.window_size:
                self.detector.fit(np.array(self.detector.raw_data_window))
        else:
            if self.detector.detect(point):
                print(f'Anomaly detected at: {time_point}')
                self.anomalies.append(time_point)
        
        return filled_point

    def get_anomalies(self):
        """
        Get the list of detected anomalies.

        Returns:
        list: List of time points where anomalies were detected.
        """
        return self.anomalies

    def get_missing_value_points(self):
        """
        Get the list of data points with handled missing values.

        Returns:
        list: List of tuples containing time points and processed data points with handled missing values.
        """
        return self.missing_value_points
