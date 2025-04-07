"""
Feature engineering utilities for the NIDS project.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE

# Initialize logger
logger = get_logger('feature_engineering')

def preprocess_data(df, label_column='Label', categorical_columns=None, drop_columns=None):
    """
    Preprocess the dataset for training.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        label_column (str): Name of the label column
        categorical_columns (list): List of categorical columns
        drop_columns (list): List of columns to drop
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    logger.info("Preprocessing data")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Default values
    if categorical_columns is None:
        categorical_columns = []
    
    if drop_columns is None:
        drop_columns = []
    
    # Drop specified columns
    if drop_columns:
        df = df.drop(columns=drop_columns, errors='ignore')
        logger.info(f"Dropped columns: {drop_columns}")
    
    # Handle missing values
    logger.info("Handling missing values")
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Separate features and labels
    X = df.drop(columns=[label_column], errors='ignore')
    y = df[label_column] if label_column in df.columns else None
    
    if y is None:
        logger.warning(f"Label column '{label_column}' not found in DataFrame")
        return None
    
    # Extract feature names
    feature_names = X.columns.tolist()
    
    # Encode categorical features
    for col in categorical_columns:
        if col in X.columns:
            logger.info(f"Encoding categorical column: {col}")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle IP addresses - convert to numeric
    for col in X.columns:
        if 'ip' in col.lower() and X[col].dtype == 'object':
            logger.info(f"Converting IP column to numeric: {col}")
            X[col] = X[col].apply(ip_to_int)
    
    # Ensure all data is numeric
    X = X.select_dtypes(include=['number'])
    logger.info(f"Final feature count: {X.shape[1]}")
    
    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, 
        random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Scale features
    logger.info("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Check class distribution
    class_counts = y_train.value_counts()
    logger.info(f"Class distribution in training set: {class_counts.to_dict()}")
    
    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        feature_names, scaler
    )

def balance_classes(X, y, method='oversample', sampling_strategy='auto'):
    """
    Balance class distribution in the dataset.
    
    Args:
        X: Feature matrix
        y: Target vector
        method (str): 'oversample', 'undersample', or 'smote'
        sampling_strategy: Strategy for resampling
        
    Returns:
        tuple: Balanced X and y
    """
    logger.info(f"Balancing classes using {method}")
    
    # Convert to numpy arrays to avoid index issues
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        X_columns = X.columns
    else:
        X_array = X
        X_columns = None
        
    if isinstance(y, pd.Series):
        y_array = y.values
        y_name = y.name
    else:
        y_array = y
        y_name = None
    
    # Get class distribution
    unique_classes, class_counts = np.unique(y_array, return_counts=True)
    class_dist = dict(zip(unique_classes, class_counts))
    logger.info(f"Original class distribution: {class_dist}")
    
    if method == 'oversample':
        # Find the majority class count
        majority_count = max(class_counts)
        
        # Initialize arrays for balanced data
        X_balanced = []
        y_balanced = []
        
        # Process each class
        for cls in unique_classes:
            # Get indices for this class
            cls_indices = np.where(y_array == cls)[0]
            
            # Get samples of this class
            X_cls = X_array[cls_indices]
            y_cls = y_array[cls_indices]
            
            # Add original samples
            X_balanced.append(X_cls)
            y_balanced.append(y_cls)
            
            # Oversample if needed
            if len(X_cls) < majority_count:
                # Calculate how many additional samples we need
                n_samples_needed = majority_count - len(X_cls)
                
                # Generate random indices with replacement
                resample_indices = np.random.choice(
                    len(X_cls),
                    size=n_samples_needed,
                    replace=True
                )
                
                # Add resampled examples
                X_balanced.append(X_cls[resample_indices])
                y_balanced.append(y_cls[resample_indices])
        
        # Concatenate all classes
        X_balanced = np.vstack(X_balanced)
        y_balanced = np.concatenate(y_balanced)
        
    elif method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
            X_balanced, y_balanced = smote.fit_resample(X_array, y_array)
            
        except ImportError:
            logger.warning("SMOTE requires imbalanced-learn. Falling back to oversampling.")
            return balance_classes(X, y, method='oversample')
    
    else:
        # For 'undersample' or unknown methods, return original data
        logger.warning(f"Method {method} not fully implemented. Returning original data.")
        return X, y
    
    # Convert back to DataFrame/Series if input was DataFrame/Series
    if X_columns is not None:
        X_balanced = pd.DataFrame(X_balanced, columns=X_columns)
    
    if y_name is not None:
        y_balanced = pd.Series(y_balanced, name=y_name)
    
    # Get new distribution
    unique_balanced, balanced_counts = np.unique(y_balanced, return_counts=True)
    balanced_dist = dict(zip(unique_balanced, balanced_counts))
    logger.info(f"Balanced class distribution: {balanced_dist}")
    
    return X_balanced, y_balanced

def ip_to_int(ip_str):
    """
    Convert an IP address string to an integer.
    
    Args:
        ip_str (str): IP address in string format
        
    Returns:
        int: IP address as integer
    """
    try:
        # Remove any CIDR notation
        if '/' in ip_str:
            ip_str = ip_str.split('/')[0]
        
        # Handle IPv4
        if '.' in ip_str:
            octets = ip_str.split('.')
            if len(octets) != 4:
                return 0
            return sum(int(octet) * (256 ** (3 - i)) for i, octet in enumerate(octets))
        
        # Handle IPv6 (simplified)
        elif ':' in ip_str:
            return int(ip_str.replace(':', ''), 16) % (2**32)
        
        return 0
    except:
        return 0

def extract_time_features(df, timestamp_col='timestamp'):
    """
    Extract time-based features from timestamp column.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        timestamp_col (str): Name of the timestamp column
        
    Returns:
        pandas.DataFrame: DataFrame with additional time features
    """
    if timestamp_col not in df.columns:
        logger.warning(f"Timestamp column '{timestamp_col}' not found")
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    
    # Extract time features
    df['hour_of_day'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_night'] = df['hour_of_day'].apply(lambda x: 1 if x < 6 or x >= 22 else 0)
    
    return df