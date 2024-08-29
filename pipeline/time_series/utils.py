import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import STL
from scipy.fftpack import fft
from scipy.signal import get_window


def load_data(file_path, timestamp_col='None', header=None):
    """
    Load the dataset from a given file path.

    Parameters:
    file_path (str): Path to the dataset file.
    timestamp_col (str, optional): Column name to parse as timestamps. If 'None', no timestamp parsing is done. Defaults to 'None'.
    header (int or list of int, optional): Row number(s) to use as the column names. Defaults to None.

    Returns:
    pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    if timestamp_col == 'None':
        return pd.read_csv(file_path, header=header)
    else:
        return pd.read_csv(file_path, parse_dates=[timestamp_col], header=header)

def standardize_data(df, feature_columns, use_pca=False, n_components=None, scale_method='zscore'):
    """
    Standardize or scale the feature columns in the dataset and optionally apply PCA.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the features to be standardized.
    feature_columns (list): List of column names to standardize.
    use_pca (bool, optional): Whether to apply PCA for dimensionality reduction. Defaults to False.
    n_components (int, optional): Number of components for PCA. If None, all components are kept. Defaults to None.
    scale_method (str, optional): Method to scale the data. Options are 'zscore' for standardization and 'minmax' for Min-Max scaling. Defaults to 'zscore'.

    Returns:
    tuple: Transformed DataFrame (or PCA-transformed data), scaler used, and PCA object (if used, otherwise None).
    """
    df_transformed = df.copy()
    
    if scale_method == 'minmax':
        scaler = MinMaxScaler()
    elif scale_method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling type")
        
    df_transformed[feature_columns] = scaler.fit_transform(df_transformed[feature_columns])
    
    if use_pca:
        pca = PCA(n_components=n_components)
        df_pca = pca.fit_transform(df_transformed[feature_columns])
        return df_pca, scaler, pca
    else:
        return df_transformed[feature_columns], scaler, None
    
def plot_multivariate_time_series(df, feature_columns, true_anomalies=None, pred=None, title='Multivariate Time Series Data'):
    """
    Plot the multivariate time series data on the same graph with shaded anomaly periods.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    feature_columns (list): List of columns to plot.
    true_anomalies (list, optional): List of true anomaly indices or timestamps. Defaults to None.
    pred (list, optional): List of predicted anomaly indices or timestamps. Defaults to None.
    title (str, optional): Title of the plot. Defaults to 'Multivariate Time Series Data'.

    Returns:
    None: The function displays the plot.
    """
    plt.figure(figsize=(15, 8))
    
    for column in feature_columns:
        plt.plot(df.index, df[column], label=column, alpha=0.7)
    
    if true_anomalies:
        for start, end in zip(true_anomalies[:-1], true_anomalies[1:]):
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                if (end - start) == pd.Timedelta(days=1):
                    plt.axvspan(start, end, color='yellow', alpha=0.7)
            else:
                if end == start + 1:
                    plt.axvspan(df.index[start], df.index[end], color='yellow', alpha=0.7)
    
    if pred:
        for start, end in zip(pred[:-1], pred[1:]):
            if isinstance(start, pd.Timestamp) and isinstance(end, pd.Timestamp):
                if (end - start) == pd.Timedelta(days=1):
                    plt.axvspan(start, end, color='red', alpha=0.7)
            else:
                if end == start + 1:
                    plt.axvspan(df.index[start], df.index[end], color='red', alpha=0.7)

    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()
    
def plot_stl_decompose(ts, period=None, seasonal=7, robust=True):
    """
    Perform STL decomposition on a time series and plot the results.

    Parameters:
    ts (pd.Series): Time series data.
    period (int, optional): The period of the seasonal component. If None, it will be estimated using FFT. Defaults to None.
    seasonal (int, optional): The length of the seasonal smoother. Defaults to 7.
    robust (bool, optional): Whether to use a robust version of the decomposition. Defaults to True.

    Returns:
    tuple: Detrended series, deseasonalized series, and detrended and deseasonalized series.
    """
    if period is None:
        freqs = np.fft.fftfreq(len(ts))
        fft_values = np.fft.fft(ts.values)
        period = abs(freqs[np.argmax(np.abs(fft_values))])
        period = int(round(1 / period))
        print(f"Estimated period: {period}")
    
    stl = STL(ts, period=period, seasonal=seasonal, robust=robust)
    result = stl.fit()
    
    detrended = ts - result.trend
    
    deseasonalized = ts - result.seasonal
    
    detrended_deseasonalized = ts - result.trend - result.seasonal
    
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(ts, label='Original')
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(result.trend, label='Trend')
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal, label='Seasonal')
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.plot(detrended_deseasonalized, label='Residual')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return detrended, deseasonalized, detrended_deseasonalized

def analyze_combined_signal(df, window_type='hamming'):
    """
    Analyze a combined signal from multiple time series by applying a window function and computing its FFT.

    Parameters:
    df (pd.DataFrame): DataFrame containing multiple time series to be combined.
    window_type (str, optional): Type of window function to apply. Defaults to 'hamming'.

    Returns:
    None: The function displays the time-domain and frequency-domain plots.
    """
    N = len(df)
    num_variables = df.shape[1]

    combined_signal = np.zeros(N)
    for i in range(num_variables):
        combined_signal += df.iloc[:, i]

    window = get_window(window_type, N)
    windowed_signal = combined_signal * window

    yf = fft(np.array(windowed_signal))
    xf = np.fft.fftfreq(N)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(combined_signal, label='Original Time Series')
    plt.plot(windowed_signal, label='Windowed Time Series', linestyle='--')
    plt.title("Time Domain")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(xf[:N // 2], np.abs(yf[:N // 2]) / N)
    plt.title(f'Combined Signal - Window Function: {window_type}')
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude")
    plt.show()
