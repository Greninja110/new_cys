"""
Ensemble model implementation combining XGBoost and Deep Learning for NIDS.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import MODEL_DIR
from models.xgboost_model import XGBoostModel
from models.deep_model import DeepModel

# Initialize logger
logger = get_logger('ensemble')

class EnsembleModel:
    """
    Ensemble model combining XGBoost and Deep Learning for network intrusion detection.
    """
    def __init__(self, xgb_params=None, deep_params=None, weights=None, model_path=None):
        """
        Initialize the ensemble model.
        
        Args:
            xgb_params (dict, optional): XGBoost parameters
            deep_params (dict, optional): Deep Learning parameters
            weights (list, optional): Weights for combining model predictions [xgb_weight, deep_weight]
            model_path (str, optional): Path to saved model
        """
        self.xgb_model = XGBoostModel(params=xgb_params)
        self.deep_model = DeepModel(params=deep_params)
        self.weights = weights if weights is not None else [0.6, 0.4]  # Default weights
        self.feature_names = None
        self.class_names = None
        
        # Load model if path is provided
        if model_path is not None:
            self.load(model_path)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, class_names=None):
        """
        Train both models in the ensemble.
        
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
        logger.info("Training ensemble model")
        
        # Store feature and class names
        self.feature_names = feature_names
        self.class_names = class_names
        
        # Train XGBoost model
        logger.info("Training XGBoost component")
        self.xgb_model.fit(X_train, y_train, X_val, y_val, feature_names, class_names)
        
        # Train Deep Learning model
        logger.info("Training Deep Learning component")
        self.deep_model.fit(X_train, y_train, X_val, y_val, feature_names, class_names)
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Ensemble training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions using the ensemble model.
        
        Args:
            X: Input features
            threshold (float): Threshold for binary classification
            
        Returns:
            np.ndarray: Predicted classes
        """
        # Get predictions from both models
        xgb_proba = self.xgb_model.predict_proba(X)
        deep_proba = self.deep_model.predict_proba(X)
        
        # Weighted average of probabilities
        ensemble_proba = (
            self.weights[0] * xgb_proba +
            self.weights[1] * deep_proba
        )
        
        # Get class with highest probability
        y_pred_indices = np.argmax(ensemble_proba, axis=1)
        
        # Convert back to original labels
        if hasattr(self.deep_model, 'label_encoder') and self.deep_model.label_encoder is not None:
            y_pred = self.deep_model.label_encoder.inverse_transform(y_pred_indices)
        else:
            y_pred = y_pred_indices
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Class probabilities
        """
        # Get predictions from both models
        xgb_proba = self.xgb_model.predict_proba(X)
        deep_proba = self.deep_model.predict_proba(X)
        
        # Weighted average of probabilities
        ensemble_proba = (
            self.weights[0] * xgb_proba +
            self.weights[1] * deep_proba
        )
        
        return ensemble_proba
    
    def evaluate(self, X_test, y_test, output_dir=None):
        """
        Evaluate the ensemble model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir (str, optional): Directory to save evaluation plots
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating ensemble model")
        
        # Create output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # Evaluate individual models
            logger.info("Evaluating XGBoost model")
            xgb_metrics = self.xgb_model.evaluate(X_test, y_test, 
                                                  os.path.join(output_dir, 'xgboost'))
            
            logger.info("Evaluating Deep Learning model")
            deep_metrics = self.deep_model.evaluate(X_test, y_test,
                                                   os.path.join(output_dir, 'deep'))
        
        # Make ensemble predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        
        logger.info(f"Ensemble Accuracy: {accuracy:.4f}")
        logger.info("Ensemble Classification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Save plots if output directory is provided
        if output_dir is not None:
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Ensemble Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'ensemble_confusion_matrix.png'))
            
            # Plot comparison of models
            self._plot_model_comparison(
                {
                    'XGBoost': xgb_metrics['classification_report'],
                    'Deep Learning': deep_metrics['classification_report'],
                    'Ensemble': clf_report
                },
                os.path.join(output_dir, 'model_comparison.png')
            )
        
        # Return evaluation metrics
        return {
            'accuracy': accuracy,
            'classification_report': clf_report,
            'confusion_matrix': cm
        }
    
    def save(self, path=None):
        """
        Save the ensemble model.
        
        Args:
            path (str, optional): Directory path to save the model
            
        Returns:
            str: Path to saved model
        """
        # Create save directory if it doesn't exist
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(MODEL_DIR, f"ensemble_{timestamp}")
        
        os.makedirs(path, exist_ok=True)
        
        # Save individual models
        xgb_path = os.path.join(path, "xgboost_model.json")
        deep_path = os.path.join(path, "deep_model.h5")
        
        self.xgb_model.save(xgb_path)
        self.deep_model.save(deep_path)
        
        # Save ensemble metadata
        metadata_path = os.path.join(path, "ensemble_metadata.joblib")
        metadata = {
            'weights': self.weights,
            'feature_names': self.feature_names,
            'class_names': self.class_names
        }
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Ensemble model saved to {path}")
        return path
    
    def load(self, path):
        """
        Load a saved ensemble model.
        
        Args:
            path (str): Directory path to the saved model
            
        Returns:
            self: Loaded model
        """
        logger.info(f"Loading ensemble model from {path}")
        
        try:
            # Load individual models
            xgb_path = os.path.join(path, "xgboost_model.json")
            deep_path = os.path.join(path, "deep_model.h5")
            
            self.xgb_model.load(xgb_path)
            self.deep_model.load(deep_path)
            
            # Load ensemble metadata
            metadata_path = os.path.join(path, "ensemble_metadata.joblib")
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.weights = metadata.get('weights', self.weights)
                self.feature_names = metadata.get('feature_names', None)
                self.class_names = metadata.get('class_names', None)
                
                logger.info(f"Loaded weights: {self.weights}")
                logger.info(f"Loaded feature names: {self.feature_names}")
                logger.info(f"Loaded class names: {self.class_names}")
            else:
                logger.warning("Metadata file not found")
            
            logger.info("Ensemble model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
        
        return self
    
    def _plot_model_comparison(self, metrics_dict, output_path=None):
        """
        Plot comparison of models.
        
        Args:
            metrics_dict (dict): Dictionary of model metrics
            output_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Model comparison plot
        """
        # Extract metrics for each model
        model_names = list(metrics_dict.keys())
        f1_scores = []
        
        for model_name, metrics in metrics_dict.items():
            # Extract f1-scores
            f1 = [metrics['macro avg']['f1-score']]
            
            # Add per-class f1-scores
            for class_name in self.class_names:
                if class_name in metrics:
                    f1.append(metrics[class_name]['f1-score'])
            
            f1_scores.append(f1)
        
        # Create labels
        labels = ['Macro Avg'] + self.class_names
        
        # Create bar chart
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(labels))
        width = 0.2
        
        for i, model_name in enumerate(model_names):
            plt.bar(x + (i - 1) * width, f1_scores[i], width, label=model_name)
        
        plt.ylabel('F1-Score')
        plt.title('Model Comparison by F1-Score')
        plt.xticks(x, labels, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save plot if output path is provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {output_path}")
        
        return plt.gcf()