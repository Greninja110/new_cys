"""
Configuration settings for the NIDS project.
"""
import os
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
LOG_DIR = os.path.join(BASE_DIR, "logs")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, UPLOAD_FOLDER]:
    os.makedirs(directory, exist_ok=True)

# Log file configuration
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"nids_{current_time}.log")

# Training parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25  # This is 25% of the training set

# XGBoost model parameters
XGBOOST_PARAMS = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    'objective': 'multi:softprob',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',  # Using GPU acceleration
    'predictor': 'gpu_predictor',
    'random_state': RANDOM_STATE
}

# Deep learning model parameters
DEEP_LEARNING_PARAMS = {
    'lstm_units': 128,
    'dense_units': 64,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 256,
    'epochs': 50,
    'patience': 10
}

# Attack types mapping
ATTACK_TYPES = {
    'benign': 0,
    'dos': 1, 
    'ddos': 2,
    'port_scan': 3,
    'brute_force': 4,
    'web_attack': 5,
    'botnet': 6,
    'infiltration': 7,
    'heartbleed': 8
}

# Features to extract from PCAP files
PCAP_FEATURES = [
    'timestamp', 'src_ip', 'dst_ip', 'protocol', 'src_port', 'dst_port',
    'packet_length', 'tcp_flags', 'tcp_window_size', 'ttl',
    'payload_length', 'packet_count', 'flow_duration'
]

# Flask configuration
FLASK_SECRET_KEY = 'your-secret-key-here'
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload size
ALLOWED_EXTENSIONS = {'pcap', 'pcapng'}