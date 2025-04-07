"""
Train NIDS models on the dataset.
"""
import os
import sys
import pandas as pd
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import get_logger
from utils.feature_engineering import preprocess_data, balance_classes
from models.xgboost_model import XGBoostModel
from models.deep_model import DeepModel
from models.ensemble import EnsembleModel
from config import MODEL_DIR, RANDOM_STATE

# Initialize logger
logger = get_logger('train')

def train_models(data_file, label_column='Label', model_type='ensemble', 
                 output_dir=None, test_mode=False):
    """
    Train NIDS models on the dataset.
    
    Args:
        data_file (str): Path to dataset CSV file
        label_column (str): Name of the label column
        model_type (str): Type of model to train ('xgboost', 'deep', or 'ensemble')
        output_dir (str, optional): Directory to save trained models
        test_mode (bool): Whether to run in test mode (smaller dataset, fewer epochs)
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    start_time = time.time()
    logger.info(f"Starting training process for model type: {model_type}")
    
    # Set output directory
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(MODEL_DIR, f"{model_type}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logging to file as well
    log_file = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] - %(name)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        # Load data
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        
        # Test mode - use smaller dataset
        if test_mode:
            logger.info("Running in test mode with reduced dataset")
            # Sample 10% of data, stratified by label
            if label_column in df.columns:
                unique_labels = df[label_column].unique()
                sampled_df = pd.DataFrame()
                for label in unique_labels:
                    label_data = df[df[label_column] == label]
                    sample_size = min(1000, int(len(label_data) * 0.1))
                    sampled_df = pd.concat([
                        sampled_df, 
                        label_data.sample(sample_size, random_state=RANDOM_STATE)
                    ])
                df = sampled_df
            else:
                df = df.sample(min(10000, int(len(df) * 0.1)), random_state=RANDOM_STATE)
        
        # Basic data info
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check for label column
        if label_column not in df.columns:
            logger.error(f"Label column '{label_column}' not found in dataset")
            return None, None
        
        # Explore label distribution
        label_counts = df[label_column].value_counts()
        logger.info(f"Label distribution:\n{label_counts}")
        
        # Identify categorical and IP address columns
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
        
        # Identify columns to drop (e.g., timestamp, source file, etc.)
        drop_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'file' in col.lower()]
        for col in df.columns:
            if 'ip' in col.lower() and df[col].dtype == 'object':
                # Convert IP addresses to numeric if possible, otherwise drop
                try:
                    test_val = df[col].iloc[0]
                    if not any(c.isalpha() for c in str(test_val)):
                        # Keep column for conversion
                        continue
                except:
                    pass
                drop_cols.append(col)
        
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Columns to drop: {drop_cols}")
        
        # Preprocess data
        result = preprocess_data(
            df, 
            label_column=label_column,
            categorical_columns=categorical_cols,
            drop_columns=drop_cols
        )
        
        if result is None:
            logger.error("Data preprocessing failed")
            return None, None
        
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = result
        
        # Save scaler
        if scaler is not None:
            import joblib
            scaler_path = os.path.join(output_dir, 'scaler.joblib')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Handle class imbalance
        logger.info("Balancing training data classes")
        X_train_balanced, y_train_balanced = balance_classes(X_train, y_train, method='oversample')
        
        # Get class names
        if y_train.dtype == 'object' or y_train.dtype.name == 'category':
            class_names = sorted(df[label_column].unique())
        else:
            le = LabelEncoder().fit(df[label_column])
            class_names = [str(i) for i in range(len(le.classes_))]
        
        logger.info(f"Class names: {class_names}")
        
        # Train model based on type
        if model_type == 'xgboost':
            model = XGBoostModel()
            
            if test_mode:
                model.params['n_estimators'] = 50
            
            # Fix for XGBoost num_class parameter
            # Fix for XGBoost num_class parameter
            if model.params['objective'] == 'multi:softprob':
                model.params['num_class'] = len(class_names)
            
            model.fit(X_train_balanced, y_train_balanced, X_val, y_val, feature_names, class_names)
            
        elif model_type == 'deep':
            model = DeepModel()
            
            if test_mode:
                model.params['epochs'] = 10
                model.params['batch_size'] = 64
            
            model.fit(X_train_balanced, y_train_balanced, X_val, y_val, feature_names, class_names)
            
        else:  # ensemble
            model = EnsembleModel()
            
            if test_mode:
                model.xgb_model.params['n_estimators'] = 50
                model.deep_model.params['epochs'] = 10
                model.deep_model.params['batch_size'] = 64
            
            # Fix for XGBoost num_class parameter
            # Fix for XGBoost num_class parameter
            if model.xgb_model.params['objective'] == 'multi:softprob':
                model.xgb_model.params['num_class'] = len(class_names)
            
            model.fit(X_train_balanced, y_train_balanced, X_val, y_val, feature_names, class_names)
        
        # Evaluate model
        eval_dir = os.path.join(output_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        metrics = model.evaluate(X_test, y_test, eval_dir)
        
        # Save model
        model_path = model.save(os.path.join(output_dir, f"{model_type}_model"))
        logger.info(f"Model saved to {model_path}")
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Plot accuracy comparison if metrics available
        try:
            if metrics and 'classification_report' in metrics:
                # Extract class-wise metrics
                class_metrics = {cls: metrics['classification_report'][cls] 
                               for cls in class_names if cls in metrics['classification_report']}
                
                # Plot class-wise F1 scores
                plt.figure(figsize=(10, 6))
                classes = list(class_metrics.keys())
                f1_scores = [metrics['classification_report'][cls]['f1-score'] for cls in classes]
                
                plt.bar(classes, f1_scores)
                plt.title(f"{model_type.capitalize()} Model F1-Scores by Class")
                plt.xlabel("Class")
                plt.ylabel("F1-Score")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(eval_dir, "f1_scores_by_class.png"))
                
                # Print overall metrics
                logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            logger.warning(f"Error plotting metrics: {str(e)}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        return None, None

if __name__ == '__main__':
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Train NIDS models')
        parser.add_argument('--data', '-d', type=str, required=True,
                           help='Path to dataset CSV file')
        parser.add_argument('--label_column', '-l', type=str, default='Label',
                           help='Name of the label column')
        parser.add_argument('--model_type', '-m', type=str, default='ensemble',
                           choices=['xgboost', 'deep', 'ensemble'],
                           help='Type of model to train')
        parser.add_argument('--output_dir', '-o', type=str, default=None,
                           help='Directory to save trained models')
        parser.add_argument('--test', '-t', action='store_true',
                           help='Run in test mode with reduced dataset')
        
        # Parse arguments and train with them
        args = parser.parse_args()
        model, metrics = train_models(
            args.data, 
            args.label_column, 
            args.model_type, 
            args.output_dir,
            args.test
        )
    
    except SystemExit:
        # Use default values if command-line arguments are missing
        print("=== Running with default parameters ===")
        print("  - Looking for dataset...")
        
        # Try to find a dataset in the default locations
        data_paths = [
            "D:\\cys\\nids\\data\\processed\\CIC-DDoS2019\\DNS-testing.csv",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "CIC-DDoS2019", "DNS-testing.csv"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed", "dataset.csv")
        ]
        
        data_file = None
        for path in data_paths:
            if os.path.exists(path):
                data_file = path
                print(f"  - Found dataset: {data_file}")
                break
        
        if data_file is None:
            print("  - ERROR: No dataset found. Please provide a dataset using --data argument")
            print("  - Example: python train.py --data D:\\cys\\nids\\data\\processed\\CIC-DDoS2019\\DNS-testing.csv")
            sys.exit(1)
            
        # Set other default parameters
        label_column = "Label"
        model_type = "ensemble"
        output_dir = None  # Will generate a timestamped directory
        test_mode = False   # Use test mode by default for faster training
        
        print(f"  - Model type: {model_type}")
        print(f"  - Test mode: {test_mode}")
        print("=== Starting training ===")
        
        # Train the model with default parameters
        model, metrics = train_models(
            data_file,
            label_column,
            model_type,
            output_dir,
            test_mode
        )