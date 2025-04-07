# NIDS Usage Guide

This guide provides detailed instructions on how to use the AI-Powered Network Intrusion Detection System (NIDS).

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Web Interface](#web-interface)
5. [Command Line Usage](#command-line-usage)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Getting Started

### Initial Setup

After installing the required dependencies, set up the environment:

```bash
python run.py setup
```

This will create all necessary directories for the application.

### Quick Start

For a quick demonstration:

1. Convert sample data:
   ```bash
   python run.py convert --input sample_data --output data/processed
   ```

2. Train a test model:
   ```bash
   python run.py train --data data/processed/sample.csv --model-type xgboost --test
   ```

3. Run the web application:
   ```bash
   python run.py run
   ```

4. Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

## Data Preparation

### Converting Parquet Files

The CIC datasets are provided in parquet format. Convert them to CSV:

```bash
python run.py convert --input /path/to/parquet/files --output data/processed --recursive
```

To merge multiple CSV files:

```bash
python run.py convert --input data/processed --merge data/merged_dataset.csv --recursive
```

### Working with PCAP Files

To preprocess PCAP files directly:

```bash
python run.py preprocess --pcap-dir /path/to/pcap/files --output-dir data/processed
```

### Data Combining and Sampling

Combine multiple preprocessed datasets:

```bash
python run.py preprocess --combine data/processed/file1.csv data/processed/file2.csv --output-dir data/processed/combined.csv
```

For large datasets, use sampling:

```bash
python run.py preprocess --combine data/processed/file1.csv data/processed/file2.csv --output-dir data/processed/combined.csv --sample-size 10000
```

## Model Training

### Training Models

Train a model using the preprocessed dataset:

```bash
python run.py train --data data/processed/dataset.csv --model-type ensemble --output-dir models/saved/my_model
```

Model types:
- `xgboost`: Fast training, good for tabular data
- `deep`: Better for complex patterns, longer training time
- `ensemble`: Best overall performance, combines both approaches

### Training Options

For testing with a smaller dataset:

```bash
python run.py train --data data/processed/dataset.csv --model-type xgboost --test
```

To specify a custom label column:

```bash
python run.py train --data data/processed/dataset.csv --model-type ensemble --label-column attack_type
```

## Web Interface

### Upload and Analysis

1. Open your browser and navigate to [http://localhost:5000](http://localhost:5000)
2. Click the "Choose File" button and select a PCAP file
3. Click "Analyze Network Traffic"
4. View the results in the detailed analysis page

### Dashboard

The dashboard provides an overview of all analyses performed:

1. Go to [http://localhost:5000/dashboard](http://localhost:5000/dashboard)
2. View summary information about analyzed files
3. Click on individual analyses to see details

### Model Management

Manage trained models via the Models page:

1. Go to [http://localhost:5000/models](http://localhost:5000/models)
2. View all available models
3. Load a different model for analysis
4. View evaluation metrics for trained models

## Command Line Usage

### Analyzing PCAP Files

Analyze a PCAP file from the command line:

```bash
python run.py analyze --file path/to/capture.pcap --model models/saved/my_model/ensemble_model --scaler models/saved/my_model/scaler.joblib
```

To save results to a CSV file:

```bash
python run.py analyze --file path/to/capture.pcap --model models/saved/my_model/ensemble_model --output results.csv
```

### Running the Application

Start the web application with custom settings:

```bash
python run.py run --host 127.0.0.1 --port 8080 --debug
```

## Advanced Usage

### Customizing the Training Process

To modify the model hyperparameters, edit the `config.py` file:

```python
# XGBoost model parameters
XGBOOST_PARAMS = {
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 200,
    ...
}

# Deep learning model parameters
DEEP_LEARNING_PARAMS = {
    'lstm_units': 128,
    'dense_units': 64,
    'dropout_rate': 0.3,
    ...
}
```

### Feature Engineering

To create custom features, modify the feature extraction process in `utils/feature_engineering.py`:

1. Add new feature extraction functions
2. Update the `preprocess_data` function to include your custom features

### Using the API

Access the NIDS via a simple API interface:

```bash
curl -X POST -F "file=@/path/to/your.pcap" http://localhost:5000/api/predict
```

The API returns a JSON object with the analysis results:

```json
{
  "filename": "example.pcap",
  "total_flows": 1250,
  "benign_count": 1000,
  "malicious_count": 250,
  "malicious_percent": 20.0,
  "prediction_counts": {
    "benign": 1000,
    "dos": 150,
    "port_scan": 100
  }
}
```

## Troubleshooting

### Common Issues

1. **Error: "No module named 'package'"**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Error converting PCAP file**
   - Check if the PCAP file is valid: `tcpdump -r file.pcap`
   - Try using pyshark instead of dpkt: Edit `pcap_utils.py` to use pyshark

3. **Model training fails or crashes**
   - Reduce batch size in `config.py`
   - Check GPU memory usage with `nvidia-smi`
   - Try training on a subset of the data with `--test` flag

4. **Flask application won't start**
   - Check if port 5000 is already in use: `netstat -ano | findstr 5000`
   - Use a different port: `python run.py run --port 8080`

### Log Files

Check log files for detailed error information:

```bash
tail -f logs/nids_YYYYMMDD_HHMMSS.log
```

Log levels can be adjusted in `utils/logger.py`.

### GPU Troubleshooting

For GPU-related issues:

1. Check if TensorFlow can see your GPU:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

2. Verify GPU drivers are compatible with your TensorFlow version
3. Monitor GPU usage during training: `nvidia-smi -l 1`

### Getting Help

For additional assistance:

1. Check the documentation in the `docs/` directory
2. Review code comments for function details
3. Create a detailed issue report if you encounter persistent problems