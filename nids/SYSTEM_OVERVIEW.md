# AI-Powered NIDS System Overview

This document provides a technical overview of the Network Intrusion Detection System architecture, components, and data flow.

## Architecture Overview

The NIDS follows a modular architecture, separating concerns into distinct components:

```
+----------------+     +----------------+     +----------------+
|                |     |                |     |                |
|  Data Pipeline |---->|  AI Models     |---->|  Web Interface |
|                |     |                |     |                |
+----------------+     +----------------+     +----------------+
       ^                      |                      |
       |                      v                      v
+----------------+     +----------------+     +----------------+
|                |     |                |     |                |
|  PCAP Analysis |     |  Prediction    |     |  Visualization |
|                |     |  Engine        |     |  & Reporting   |
+----------------+     +----------------+     +----------------+
```

## Components

### 1. Data Pipeline

The data pipeline handles the conversion, preprocessing, and feature extraction from network traffic data:

- **parquet_to_csv.py**: Converts parquet dataset files to CSV format
- **preprocess.py**: Processes raw data into a format suitable for model training
- **utils/pcap_utils.py**: Extracts features from PCAP files
- **utils/feature_engineering.py**: Creates and transforms features for the models

### 2. AI Models

Three model types are implemented, each with specific advantages:

#### XGBoost Model (`models/xgboost_model.py`)

- Gradient boosting framework optimized for tabular data
- Fast training and inference speeds
- Good interpretability via feature importance

```python
class XGBoostModel:
    def __init__(self, params=None, model_path=None)
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, class_names=None)
    def predict(self, X, threshold=0.5)
    def evaluate(self, X_test, y_test, output_dir=None)
    def save(self, path=None)
    def load(self, path)
```

#### Deep Learning Model (`models/deep_model.py`)

- LSTM-based neural network for sequence modeling
- Better for capturing temporal patterns
- Higher capacity for complex relationships

```python
class DeepModel:
    def __init__(self, params=None, model_path=None)
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, class_names=None)
    def predict(self, X, threshold=0.5)
    def evaluate(self, X_test, y_test, output_dir=None)
    def save(self, path=None)
    def load(self, path)
```

#### Ensemble Model (`models/ensemble.py`)

- Combines predictions from both XGBoost and Deep Learning models
- Weighted averaging for optimal performance
- Higher accuracy and robustness

```python
class EnsembleModel:
    def __init__(self, xgb_params=None, deep_params=None, weights=None, model_path=None)
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, class_names=None)
    def predict(self, X, threshold=0.5)
    def evaluate(self, X_test, y_test, output_dir=None)
    def save(self, path=None)
    def load(self, path)
```

### 3. Web Interface

The Flask-based web interface provides user interaction:

- **app.py**: Main Flask application with routes and logic
- **templates/**: HTML templates for different pages
- **static/**: CSS, JavaScript, and image assets

Key routes:
- `/`: Main page for file upload
- `/upload`: Handles file upload and processing
- `/dashboard`: Displays summary of analyses
- `/models`: Model management interface
- `/about`: Information about the system
- `/api/predict`: API endpoint for programmatic access

### 4. Configuration and Utilities

Supporting components:

- **config.py**: Central configuration for all components
- **utils/logger.py**: Logging utility for consistent logging
- **run.py**: Command-line interface for all functions

## Data Flow

### Training Flow

```
Raw Datasets (Parquet)
       |
       v
Convert to CSV (parquet_to_csv.py)
       |
       v
Preprocess Data (preprocess.py)
       |
       v
Extract Features (feature_engineering.py)
       |
       v
Train Models (train.py)
       |
       v
Save Models and Metrics
```

### Inference Flow

```
User Upload PCAP --> Extract Features --> Model Prediction --> Generate Visualizations --> Display Results
```

### API Flow

```
Client Request --> Parse PCAP --> Extract Features --> Model Prediction --> JSON Response
```

## Communication Between Components

Components communicate through well-defined interfaces:

1. **Data Passing**: Through pandas DataFrames and numpy arrays
2. **Model Serialization**: Via model-specific save/load methods
3. **Web Communication**: HTTP requests/responses with Flask
4. **Configuration**: Centralized in config.py for consistency

## Technology Stack

- **Python 3.12.8**: Core programming language
- **Data Processing**: pandas, numpy, scikit-learn
- **AI/ML**: TensorFlow, XGBoost
- **PCAP Processing**: dpkt, pyshark, scapy
- **Visualization**: plotly, matplotlib, seaborn
- **Web Framework**: Flask
- **Front-end**: Bootstrap, jQuery, Plotly.js

## Key Design Patterns

1. **Factory Pattern**: For model creation and selection
2. **Strategy Pattern**: For different preprocessing strategies
3. **Adapter Pattern**: For handling different file formats
4. **Facade Pattern**: Simplified interface to complex subsystems
5. **Observer Pattern**: For logging and monitoring

## Performance Considerations

- **GPU Acceleration**: Used for both XGBoost and TensorFlow
- **Batch Processing**: For handling large files
- **Caching**: Intermediate results saved to disk
- **Parallel Processing**: Where applicable for feature extraction

## Security Considerations

- **Input Validation**: All user inputs are validated
- **File Size Limits**: Prevents DoS attacks
- **Secure File Handling**: Safe file naming and storage
- **Error Handling**: Prevents information leakage

## Extending the System

### Adding New Model Types

1. Create a new model class in the `models/` directory
2. Implement the required interface (fit, predict, evaluate, save, load)
3. Update the model factory in `train.py`
4. Add the new model option to the web interface

### Adding New Attack Types

1. Update the `ATTACK_TYPES` dictionary in `config.py`
2. Ensure training data includes examples of the new attack type
3. Retrain models with the updated dataset

### Custom Feature Extraction

1. Add new feature extraction functions to `utils/feature_engineering.py`
2. Update the preprocessing pipeline in `preprocess.py`
3. Document the new features for future reference

## System Requirements and Scaling

### Minimum Requirements

- **CPU**: Intel Core i5 or equivalent
- **RAM**: 16GB
- **Storage**: 10GB for application, 50GB+ for datasets
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional but recommended)

### Scaling Considerations

- **Large Datasets**: Use sampling or incremental learning
- **Multiple Users**: Implement job queue for analysis requests
- **Real-time Analysis**: Consider streaming data processing
- **High Availability**: Deploy multiple instances behind load balancer