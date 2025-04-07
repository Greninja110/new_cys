"""
XGBoost model implementation for NIDS.
"""
import os
import sys
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import XGBOOST_PARAMS, MODEL_DIR

# Initialize logger
logger = get_logger('xgboost_model')

class XGBoostModel:
    """
    XGBoost model for network intrusion detection.
    """
    def __init__(self, params=None, model_path=None):
        """
        Initialize the XGBoost model.
        
        Args:
            params (dict, optional): XGBoost parameters
            model_path (str, optional): Path to saved model
        """
        self.params = params if params is not None else XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.class_names = None
        
        # Load model if path is provided
        if model_path is not None:
            self.load(model_path)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, class_names=None):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names (list, optional): Feature names
            class_names (list, optional): Class names
            
        Returns:
            self: Trained model
        """
        start_time = time.time()
        logger.info("Training XGBoost model")
        
        # Store feature and class names
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Check if labels are strings or categorical and encode them
        if y_train.dtype == 'object' or y_train.dtype.name == 'category':
            logger.info("Converting string labels to numeric values")
            self.label_encoder = LabelEncoder()
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            
            # Map the classes for reference
            self.label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
            logger.info(f"Label mapping: {self.label_mapping}")
            
            if y_val is not None:
                y_val_encoded = self.label_encoder.transform(y_val)
            else:
                y_val_encoded = None
        else:
            # Labels are already numeric
            self.label_encoder = None
            y_train_encoded = y_train
            y_val_encoded = y_val
        
        # Prepare data
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded, feature_names=feature_names)
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            threshold (float): Threshold for binary classification
            
        Returns:
            np.ndarray: Predicted classes
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Make predictions
        if self.params.get('num_class', 0) > 2:
            # Multi-class prediction - return class with highest probability
            probs = self.model.predict(dtest)
            predictions = np.argmax(probs, axis=1)
        else:
            # Binary classification
            probs = self.model.predict(dtest)
            predictions = (probs > threshold).astype(int)
        
        # Convert back to original labels if a label encoder was used
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(predictions)
        else:
            return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Class probabilities
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        
        # Make predictions
        if self.params.get('num_class', 0) > 2:
            # Multi-class prediction
            return self.model.predict(dtest)
        else:
            # Binary classification - return [1-p, p]
            probs = self.model.predict(dtest)
            return np.vstack((1 - probs, probs)).T
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir (str, optional): Directory to save evaluation plots
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        logger.info("Evaluating XGBoost model")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save plots if output directory is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            
            # Plot feature importance
            self.plot_feature_importance(os.path.join(output_dir, 'feature_importance.png'))
        
        # Return evaluation metrics
        return {
            'accuracy': accuracy,
            'classification_report': clf_report,
            'confusion_matrix': cm
        }
    
    def save(self, path=None):
        """
        Save the trained model.
        
        Args:
            path (str, optional): Path to save the model
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            logger.error("Cannot save: Model not trained")
            return None
        
        # Create save directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Generate a path if not provided
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(MODEL_DIR, f"xgboost_model_{timestamp}.json")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save_model(path)
        
        # Save metadata
        metadata_path = os.path.splitext(path)[0] + "_metadata.joblib"
        metadata = {
            'params': self.params,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'label_encoder': getattr(self, 'label_encoder', None),
            'label_mapping': getattr(self, 'label_mapping', None)
        }
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def load(self, path):
        """
        Load a saved model.
        
        Args:
            path (str): Path to saved model
            
        Returns:
            self: Loaded model
        """
        logger.info(f"Loading model from {path}")
        
        # Create model instance
        self.model = xgb.Booster()
        
        try:
            # Load model
            self.model.load_model(path)
            
            # Load metadata
            metadata_path = os.path.splitext(path)[0] + "_metadata.joblib"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.params = metadata.get('params', self.params)
                self.feature_names = metadata.get('feature_names', None)
                self.class_names = metadata.get('class_names', None)
                self.label_encoder = metadata.get('label_encoder', None)
                self.label_mapping = metadata.get('label_mapping', None)
                
                logger.info(f"Loaded metadata: {self.params}")
                logger.info(f"Loaded feature names: {self.feature_names}")
                logger.info(f"Loaded class names: {self.class_names}")
                if self.label_encoder is not None:
                    logger.info(f"Loaded label mapping: {self.label_mapping}")
            else:
                logger.warning("Metadata file not found")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
        
        return self
    
    def plot_feature_importance(self, output_path=None):
        """
        Plot feature importance.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Feature importance plot
        """
        if self.model is None:
            logger.error("Model not trained or loaded")
            return None
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        
        # If no importance found
        if not importance:
            logger.warning("No feature importance found")
            return None
        
        # Sort features by importance
        sorted_idx = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_idx]
        values = [x[1] for x in sorted_idx]
        
        # Limit to top 20 features if there are more
        if len(features) > 20:
            features = features[:20]
            values = values[:20]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance (Gain)')
        plt.tight_layout()
        
        # Save plot if output path is provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        
        return plt.gcf()