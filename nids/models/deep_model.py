"""
Deep learning model implementation for NIDS.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import DEEP_LEARNING_PARAMS, MODEL_DIR, RANDOM_STATE

# Initialize logger
logger = get_logger('deep_model')

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class DeepModel:
    """
    Deep learning model for network intrusion detection.
    """
    def __init__(self, params=None, model_path=None):
        """
        Initialize the deep learning model.
        
        Args:
            params (dict, optional): Model parameters
            model_path (str, optional): Path to saved model
        """
        self.params = params if params is not None else DEEP_LEARNING_PARAMS.copy()
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.class_names = None
        self.history = None
        
        # Configure GPU memory growth to avoid OOM errors
        self._configure_gpu()
        
        # Load model if path is provided
        if model_path is not None:
            self.load(model_path)
    
    def _configure_gpu(self):
        """Configure GPU settings for TensorFlow."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPUs")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU memory growth enabled for {gpu}")
            else:
                logger.info("No GPUs found, using CPU")
        except Exception as e:
            logger.warning(f"Error configuring GPU: {e}")
    
    def _create_model(self, input_dim, num_classes):
        """
        Create the deep learning model architecture.
        
        Args:
            input_dim (int): Input dimension
            num_classes (int): Number of classes
            
        Returns:
            tf.keras.Model: Created model
        """
        logger.info(f"Creating model with input_dim={input_dim}, num_classes={num_classes}")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.params['lstm_units'],
            input_shape=(input_dim, 1),
            return_sequences=True,
            kernel_regularizer=l2(0.001)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))
        
        # Second LSTM layer
        model.add(LSTM(
            units=self.params['lstm_units'] // 2,
            return_sequences=False,
            kernel_regularizer=l2(0.001)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))
        
        # Dense layers
        model.add(Dense(
            units=self.params['dense_units'],
            activation='relu',
            kernel_regularizer=l2(0.001)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.params['dropout_rate']))
        
        # Output layer
        if num_classes > 2:
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        else:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.params['learning_rate']),
            loss=loss,
            metrics=['accuracy']
        )
        
        # Log model summary
        model.summary(print_fn=logger.info)
        
        return model
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, class_names=None):
        """
        Train the deep learning model.
        
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
        logger.info("Training deep learning model")
        
        # Store feature and class names
        self.feature_names = feature_names
        
        # Prepare labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Store class names
        if class_names is None:
            self.class_names = [str(c) for c in self.label_encoder.classes_]
        else:
            self.class_names = class_names
            
        logger.info(f"Class mapping: {dict(zip(self.class_names, range(len(self.class_names))))}")
        
        # Get number of classes
        num_classes = len(self.class_names)
        
        # Convert labels to categorical if multiclass
        if num_classes > 2:
            y_train_cat = to_categorical(y_train_encoded)
            if X_val is not None and y_val is not None:
                y_val_encoded = self.label_encoder.transform(y_val)
                y_val_cat = to_categorical(y_val_encoded)
            else:
                y_val_cat = None
        else:
            y_train_cat = y_train_encoded
            if X_val is not None and y_val is not None:
                y_val_cat = self.label_encoder.transform(y_val)
            else:
                y_val_cat = None
        
        # Reshape input for LSTM [samples, timesteps, features]
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        if X_val is not None:
            X_val_reshaped = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        else:
            X_val_reshaped = None
        
        # Create model
        self.model = self._create_model(X_train.shape[1], num_classes)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=self.params['patience'],
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=self.params['patience'] // 2,
                min_lr=1e-6
            )
        ]
        
        # Add model checkpoint if saving
        checkpoint_path = os.path.join(MODEL_DIR, 'checkpoints', 'deep_model_checkpoint.h5')
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        callbacks.append(
            ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        )
        
        # Train model
        validation_data = (X_val_reshaped, y_val_cat) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            X_train_reshaped, y_train_cat,
            validation_data=validation_data,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            callbacks=callbacks,
            verbose=2
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            threshold (float): Threshold for binary classification
            
        Returns:
            np.ndarray: Predicted classes (original labels)
        """
        if self.model is None or self.label_encoder is None:
            logger.error("Model not trained or loaded")
            return None
        
        # Reshape input for LSTM [samples, timesteps, features]
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Get raw predictions
        y_pred_raw = self.model.predict(X_reshaped)
        
        # Convert to class indices
        if y_pred_raw.shape[1] > 1:  # Multi-class
            y_pred_indices = np.argmax(y_pred_raw, axis=1)
        else:  # Binary
            y_pred_indices = (y_pred_raw > threshold).astype(int).ravel()
        
        # Convert back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_indices)
        
        return y_pred
    
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
        
        # Reshape input for LSTM [samples, timesteps, features]
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Get raw predictions
        y_pred_raw = self.model.predict(X_reshaped)
        
        # Binary case - return [1-p, p]
        if y_pred_raw.shape[1] == 1:
            probs = y_pred_raw.ravel()
            return np.vstack((1 - probs, probs)).T
        
        return y_pred_raw
    
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
        if self.model is None or self.label_encoder is None:
            logger.error("Model not trained or loaded")
            return None
        
        logger.info("Evaluating deep learning model")
        
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
            plt.savefig(os.path.join(output_dir, 'deep_confusion_matrix.png'))
            
            # Plot training history
            if self.history is not None:
                self._plot_training_history(os.path.join(output_dir, 'training_history.png'))
        
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
            path = os.path.join(MODEL_DIR, f"deep_model_{timestamp}.h5")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        
        # Save metadata
        metadata_path = os.path.splitext(path)[0] + "_metadata.joblib"
        metadata = {
            'params': self.params,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'label_encoder': self.label_encoder
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
        
        try:
            # Load model
            self.model = load_model(path)
            
            # Load metadata
            metadata_path = os.path.splitext(path)[0] + "_metadata.joblib"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.params = metadata.get('params', self.params)
                self.feature_names = metadata.get('feature_names', None)
                self.class_names = metadata.get('class_names', None)
                self.label_encoder = metadata.get('label_encoder', None)
                
                logger.info(f"Loaded metadata: {self.params}")
                logger.info(f"Loaded feature names: {self.feature_names}")
                logger.info(f"Loaded class names: {self.class_names}")
            else:
                logger.warning("Metadata file not found")
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
        
        return self
    
    def _plot_training_history(self, output_path=None):
        """
        Plot training history.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Training history plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return None
        
        # Create plot
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Training')
        if 'val_accuracy' in self.history.history:
            plt.plot(self.history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Training')
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot if output path is provided
        if output_path is not None:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {output_path}")
        
        return plt.gcf()