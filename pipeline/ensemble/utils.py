import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import IsolationForest
import shap
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st


def preprocess_and_split_data(file_path, 
                              label_header, 
                              test_size=0.2, 
                              val_size=None, 
                              random_state=42, 
                              feature_select=False, 
                              n_features=15, 
                              use_vif=True, 
                              vif_threshold=5.0, 
                              upsample=False, 
                              use_autoencoder=False, 
                              code_size=10):
    """
    Preprocess the data by optionally applying an autoencoder for feature reconstruction or selecting features, 
    then splitting into train and test sets, removing correlated features using VIF, adding Mahalanobis distance, 
    and scaling the features.

    Parameters:
    file_path (str): Path to the data file.
    label_header (str): Name of the column to be used as the label.
    test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
    val_size (float, optional): Proportion of the training data to include in the validation (meta-training) split for stacking. Defaults to None.
    random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    feature_select (bool, optional): Whether to perform feature selection. Defaults to False.
    n_features (int, optional): Number of features to select if feature_select is True. Defaults to 15.
    use_vif (bool, optional): Whether to use VIF for feature selection. Defaults to True.
    vif_threshold (float, optional): VIF threshold for feature selection if use_vif is True. Defaults to 5.0.
    upsample (bool, optional): Whether to upsample the minority class in the training set. Defaults to False.
    use_autoencoder (bool, optional): Whether to use an autoencoder for feature reconstruction. Defaults to False.
    code_size (int, optional): Size of the encoded (latent) layer for the autoencoder if use_autoencoder is True. Defaults to 10.

    Returns:
    If val_size is None:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test
    If val_size is provided:
        tuple: X_train_main_scaled, X_train_meta_scaled, X_test_scaled, y_train_main, y_train_meta, y_test
    """
    X, y = load_data(file_path, label_header)
    
    if use_autoencoder:
        result = apply_autoencoder(X, y, test_size=test_size, val_size=val_size, random_state=random_state, code_size=code_size, epochs=50, batch_size=64, learning_rate=0.001)
    else:
        if feature_select:
            X_selected = select_features(X, test_size=0.2, random_state=42, n_features=n_features)
            result = preprocess_data_with_distance(X_selected, y, test_size=test_size, val_size=val_size, random_state=random_state, use_vif=use_vif, vif_threshold=vif_threshold, upsample=upsample)
        else:
            result = preprocess_data_with_distance(X, y, test_size=test_size, val_size=val_size, random_state=random_state, use_vif=use_vif, vif_threshold=vif_threshold, upsample=upsample)
    return result

def preprocess_data_with_distance(X, y, test_size=0.2, val_size=None, random_state=42, use_vif=True, vif_threshold=5.0, upsample=False):
    """
    Preprocess the data by splitting it into train and test sets, optionally upsampling the minority class, 
    optionally removing correlated features using VIF, adding Mahalanobis distance, and standardizing the features.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
    val_size (float, optional): Proportion of the training data to include in the validation (meta-training) split for stacking. Defaults to None.
    random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    use_vif (bool, optional): Whether to use VIF for feature selection. Defaults to True.
    vif_threshold (float, optional): VIF threshold for feature selection if use_vif is True. Defaults to 5.0.
    upsample (bool, optional): Whether to upsample the minority class in the training set. Defaults to False.

    Returns:
    If val_size is None:
        tuple: X_train_scaled, X_test_scaled, y_train, y_test
    If val_size is provided:
        tuple: X_train_main_scaled, X_train_meta_scaled, X_test_scaled, y_train_main, y_train_meta, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    if val_size is not None:
        X_train_main, X_train_meta, y_train_main, y_train_meta = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
        
        if upsample:
            X_train_main, y_train_main = apply_smote(X_train_main, y_train_main, random_state=random_state)
        
        if use_vif:
            selected_features = select_features_vif(pd.DataFrame(X_train_main), threshold=vif_threshold)
            X_train_main = pd.DataFrame(X_train_main, columns=selected_features)
            X_train_meta = pd.DataFrame(X_train_meta, columns=selected_features)
            X_test = pd.DataFrame(X_test, columns=selected_features)
        else:
            selected_features = X.columns
        
        df_train_main = pd.DataFrame(X_train_main, columns=selected_features)
        df_train_meta = pd.DataFrame(X_train_meta, columns=selected_features)
        df_test = pd.DataFrame(X_test, columns=selected_features)
        
        df_train_main, df_test, df_train_meta = add_mahalanobis_distance(df_train_main, df_test, df_train_meta)
        
        scaler = StandardScaler()
        X_train_main_scaled = scaler.fit_transform(df_train_main.drop(columns=['mahalanobis']))
        X_train_meta_scaled = scaler.transform(df_train_meta.drop(columns=['mahalanobis']))
        X_test_scaled = scaler.transform(df_test.drop(columns=['mahalanobis']))
        
        X_train_main_scaled = np.hstack((X_train_main_scaled, df_train_main[['mahalanobis']].values))
        X_train_meta_scaled = np.hstack((X_train_meta_scaled, df_train_meta[['mahalanobis']].values))
        X_test_scaled = np.hstack((X_test_scaled, df_test[['mahalanobis']].values))
        
        return X_train_main_scaled, X_train_meta_scaled, X_test_scaled, y_train_main, y_train_meta, y_test
    
    else:
        if upsample:
            X_train, y_train = apply_smote(X_train, y_train, random_state=random_state)
        
        if use_vif:
            selected_features = select_features_vif(pd.DataFrame(X_train), threshold=vif_threshold)
            X_train = pd.DataFrame(X_train, columns=selected_features)
            X_test = pd.DataFrame(X_test, columns=selected_features)
        else:
            selected_features = X.columns
        
        df_train = pd.DataFrame(X_train, columns=selected_features)
        df_test = pd.DataFrame(X_test, columns=selected_features)
        
        df_train, df_test = add_mahalanobis_distance(df_train, df_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(df_train.drop(columns=['mahalanobis']))
        X_test_scaled = scaler.transform(df_test.drop(columns=['mahalanobis']))
        
        X_train_scaled = np.hstack((X_train_scaled, df_train[['mahalanobis']].values))
        X_test_scaled = np.hstack((X_test_scaled, df_test[['mahalanobis']].values))
        
        return X_train_scaled, X_test_scaled, y_train, y_test

def load_data(file_path, label_header):
    """
    Load and preprocess data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file containing the data.
    label_header (str): The header/name of the label/target column in the CSV file.

    Returns:
    X (pd.DataFrame): The feature matrix containing all columns except the label column.
    y (pd.Series): The target vector containing the label column.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    X = df.drop(label_header, axis=1)
    y = df[label_header]
    return X, y

def select_features(X, test_size=0.2, random_state=42, n_features=15):
    """
    Preprocess the data by splitting it into train and test sets and performing feature selection based on the train set. 
    Features are selected based on feature importance from Isolation Forest, and the selected features are recombined 
    into a whole matrix in the original order of X after processing.

    Parameters:
    X (array-like): Feature matrix.
    test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
    random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    n_features (int, optional): Number of features to select. Defaults to 15.

    Returns:
    X_combined (pd.DataFrame): The recombined feature matrix in original order.
    """
    X_train, X_test, train_idx, test_idx = train_test_split(X, range(len(X)), test_size=test_size, random_state=random_state)

    if X_train.shape[1] <= n_features:
        selected_indices = list(range(X_train.shape[1]))
    else:
        est = IsolationForest(random_state=random_state)
        est.fit(X_train)
        
        # Calculate SHAP values: https://github.com/shap/shap
        explainer = shap.TreeExplainer(est)
        shap_values = explainer.shap_values(X_train)
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        selected_indices = np.argsort(shap_importance)[::-1][:n_features]

    X_train_reduced = X_train.iloc[:, selected_indices]
    X_test_reduced = X_test.iloc[:, selected_indices]

    combined_data = pd.concat([X_train_reduced, X_test_reduced], axis=0)
    combined_data.index = np.concatenate([train_idx, test_idx])
    X_combined = combined_data.sort_index()

    return X_combined

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame.

    Parameters:
    X (pd.DataFrame): The feature matrix.

    Returns:
    vif (pd.DataFrame): A DataFrame containing the VIF values for each feature.
    """
    X = pd.DataFrame(X)
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


def select_features_vif(X, threshold=5.0):
    """
    Select features based on a VIF threshold.

    Parameters:
    X (pd.DataFrame): The feature matrix.
    threshold (float, optional): VIF threshold to select features. Defaults to 5.0.

    Returns:
    selected_features (list): List of selected feature names.
    """
    vif = calculate_vif(X)
    selected_features = vif[vif["VIF"] < threshold]["feature"].tolist()
    return selected_features


def add_mahalanobis_distance(df_train, df_test, df_val=None):
    """
    Adds a column of Mahalanobis distance to the training and testing dataframes as a new feature.

    Parameters:
    df_train (pd.DataFrame): Training data.
    df_test (pd.DataFrame): Testing data.
    df_val (pd.DataFrame, optional): Validation data. Defaults to None.

    Returns:
    If df_val is None:
        tuple: Updated df_train, df_test
    If df_val is provided:
        tuple: Updated df_train, df_test, df_val
    """
    mean_vector = df_train.mean().values
    cov_matrix = np.cov(df_train, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    
    def mahalanobis(row):
        diff = row - mean_vector
        return np.sqrt(diff.dot(inv_cov_matrix).dot(diff.T))
    
    df_train['mahalanobis'] = df_train.apply(lambda row: mahalanobis(row.values), axis=1)
    df_test['mahalanobis'] = df_test.apply(lambda row: mahalanobis(row.values), axis=1)
    
    if df_val is not None:
        df_val['mahalanobis'] = df_val.apply(lambda row: mahalanobis(row.values), axis=1)
        return df_train, df_test, df_val
    
    return df_train, df_test

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to the training data to handle class imbalance.

    Parameters:
    X_train (array-like): Feature matrix for the training data.
    y_train (array-like): Target vector for the training data.
    random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
    tuple: The upsampled feature matrix and target vector.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

class Autoencoder(nn.Module):
    """
    An Autoencoder class for feature reconstruction with a single hidden layer.
    """
    def __init__(self, input_size, intermediate_size, code_size):
        """
        Initialize the Autoencoder model.

        Parameters:
        input_size (int): Size of the input layer.
        intermediate_size (int): Size of the intermediate hidden layer.
        code_size (int): Size of the encoded (latent) layer.
        """
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        self.code_size = code_size

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.input_size, self.intermediate_size)
        self.fc2 = nn.Linear(self.intermediate_size, self.code_size)

        self.fc3 = nn.Linear(self.code_size, self.intermediate_size)
        self.fc4 = nn.Linear(self.intermediate_size, self.input_size)

    def forward(self, x):
        """
        Forward pass through the Autoencoder.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after reconstruction.
        """
        hidden = self.fc1(x)
        hidden = self.relu(hidden)

        code = self.fc2(hidden)
        code = self.relu(code)

        hidden = self.fc3(code)
        hidden = self.relu(hidden)

        output = self.fc4(hidden)

        return output
    
    def encode(self, x):
        """
        Encode the input data to the latent space.

        Parameters:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Latent code.
        """
        hidden = self.fc1(x)
        hidden = self.relu(hidden)

        code = self.fc2(hidden)
        code = self.relu(code)
        
        return code

class WeightedMSELoss(nn.Module):
    """
    Custom weighted mean squared error loss for the Autoencoder.
    """
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weights):
        weights = weights.view(-1, 1)
        loss = weights * (input - target) ** 2
        return loss.mean()

def train_autoencoder(train_data, weights, input_size, intermediate_size, code_size, epochs=50, batch_size=64, learning_rate=0.001):
    """
    Trains an Autoencoder model on the given training data.

    Parameters:
    train_data (array-like): Training data.
    weights (array-like): Sample weights.
    input_size (int): Size of the input layer.
    intermediate_size (int): Size of the intermediate hidden layer.
    code_size (int): Size of the encoded (latent) layer.
    epochs (int, optional): Number of training epochs. Defaults to 50.
    batch_size (int, optional): Batch size for training. Defaults to 64.
    learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
    Autoencoder: Trained Autoencoder model.
    """
    dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(weights, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(input_size, intermediate_size, code_size)
    criterion = WeightedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            inputs, sample_weights = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs, sample_weights)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}')

    return model

def encode_data(model, data):
    """
    Encodes the data using a trained Autoencoder model.

    Parameters:
    model (Autoencoder): Trained Autoencoder model.
    data (array-like): Data to be encoded.

    Returns:
    encoded_data (array-like): Latent representations.
    """
    with torch.no_grad():
        encoded_data = model.encode(torch.tensor(data, dtype=torch.float32)).numpy()
    return encoded_data

def compute_class_weights(y):
    """
    Computes sample weights based on class distribution.

    Parameters:
    y (array-like): Target vector.

    Returns:
    weights (array-like): Computed sample weights.
    """
    class_counts = np.bincount(y)
    total_samples = len(y)
    class_weights = {i: total_samples / count for i, count in enumerate(class_counts)}
    weights = np.array([class_weights[i] for i in y])
    return weights

def apply_autoencoder(X, y, test_size=0.2, val_size=None, random_state=42, code_size=10, epochs=50, batch_size=64, learning_rate=0.001):
    """
    Applies an Autoencoder for feature reconstruction, with an optional validation split,
    and standardizes the encoded data.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    test_size (float, optional): Proportion of the data to include in the test split. Defaults to 0.2.
    val_size (float, optional): Proportion of the training data to include in the validation (meta-training) split for stacking. Defaults to None.
    random_state (int, optional): Random seed for reproducibility. Defaults to 42.
    code_size (int, optional): Size of the encoded (latent) layer. Defaults to 10.
    epochs (int, optional): Number of training epochs. Defaults to 50.
    batch_size (int, optional): Batch size for training. Defaults to 64.
    learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.

    Returns:
    If val_size is None:
        tuple: X_train_encoded, X_test_encoded, y_train, y_test
    If val_size is provided:
        tuple: X_train_main_encoded, X_train_meta_encoded, X_test_encoded, y_train_main, y_train_meta, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    if val_size is not None:
        X_train_main, X_train_meta, y_train_main, y_train_meta = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)
        
        input_size = X_train_main.shape[1]
        intermediate_size = input_size // 2
        train_data = X_train_main.values
        weights = compute_class_weights(y_train_main)
        
        autoencoder = train_autoencoder(train_data, weights, input_size, intermediate_size, code_size, epochs, batch_size, learning_rate)
        
        X_train_main_encoded = encode_data(autoencoder, X_train_main.values)
        X_train_meta_encoded = encode_data(autoencoder, X_train_meta.values)
        X_test_encoded = encode_data(autoencoder, X_test.values)
        
        scaler = StandardScaler()
        X_train_main_encoded = scaler.fit_transform(X_train_main_encoded)
        X_train_meta_encoded = scaler.transform(X_train_meta_encoded)
        X_test_encoded = scaler.transform(X_test_encoded)
        
        return X_train_main_encoded, X_train_meta_encoded, X_test_encoded, y_train_main, y_train_meta, y_test
    
    else:
        input_size = X_train.shape[1]
        intermediate_size = input_size // 2
        train_data = X_train.values
        weights = compute_class_weights(y_train)
        
        autoencoder = train_autoencoder(train_data, weights, input_size, intermediate_size, code_size, epochs, batch_size, learning_rate)
        
        X_train_encoded = encode_data(autoencoder, X_train.values)
        X_test_encoded = encode_data(autoencoder, X_test.values)
        
        scaler = StandardScaler()
        X_train_encoded = scaler.fit_transform(X_train_encoded)
        X_test_encoded = scaler.transform(X_test_encoded)
        
        return X_train_encoded, X_test_encoded, y_train, y_test


def display_confusion_matrix_table(cm, labels=None):
    """
    Displays a confusion matrix as a table using Streamlit.

    Parameters:
    cm (array-like): Confusion matrix.
    labels (list of str, optional): List of label names. Defaults to None.
    """
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    df = pd.DataFrame(cm, index=[f"True {label}" for label in labels], columns=[f"Predicted {label}" for label in labels])
    st.table(df)

def print_metrics(rank, metric):
    """
    Print evaluation metrics for a model using Streamlit.

    Parameters:
    rank (int): Rank of the model.
    metric (dict): Dictionary containing evaluation metrics.
    """
    st.write(f"**Rank {rank}:**")
    st.write(f"**Method:** {metric['Method']}")
    st.write(f"**AUROC:** {metric['Test AUROC']:.4f}")
    st.write(f"**Averaged Precision:** {metric['Test Averaged Precision']:.4f}")
    st.write(f"**F1 Score:** {metric['Test F1 Score']:.4f}")
    st.write("**Confusion Matrix:**")
    display_confusion_matrix_table(np.array(metric['Test Confusion Matrix']), ["Inliers", "Outliers"])