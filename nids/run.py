"""
Main script to run the NIDS application.
Provides command-line interface for various functionalities.
"""
import os
import sys
import argparse
import time
import logging
from datetime import datetime

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('nids')

def setup_environment():
    """Create necessary directories if they don't exist."""
    logger.info("Setting up environment")
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static/img', exist_ok=True)

def convert_parquet_files(args):
    """Convert parquet files to CSV."""
    from parquet_to_csv import convert_all_parquet_files, merge_csv_files
    
    logger.info("Converting parquet files to CSV")
    csv_files = convert_all_parquet_files(args.input, args.output, args.recursive)
    
    if args.merge and csv_files:
        logger.info(f"Merging {len(csv_files)} CSV files")
        merge_csv_files(csv_files, args.merge, args.sample_size)

def preprocess_data(args):
    """Preprocess PCAP files or CSV data."""
    from preprocess import preprocess_pcap_directory, preprocess_labeled_data, combine_datasets
    
    if args.pcap_dir:
        logger.info(f"Preprocessing PCAP files in {args.pcap_dir}")
        preprocess_pcap_directory(args.pcap_dir, args.output_dir, args.recursive)
    
    if args.input_csv:
        logger.info(f"Preprocessing labeled data from {args.input_csv}")
        preprocess_labeled_data(args.input_csv, args.label_file, args.output_dir)
    
    if args.combine:
        logger.info(f"Combining {len(args.combine)} datasets")
        combine_datasets(args.combine, args.output_dir, args.sample_size)

def train_model(args):
    """Train a model on the dataset."""
    from train import train_models
    
    logger.info(f"Training {args.model_type} model with {args.data}")
    model, metrics = train_models(
        args.data,
        args.label_column,
        args.model_type,
        args.output_dir,
        args.test
    )
    
    if model and metrics:
        logger.info(f"Model training completed with accuracy: {metrics.get('accuracy', 'N/A')}")
        return True
    else:
        logger.error("Model training failed")
        return False

def run_flask_app(args):
    """Run the Flask web application."""
    try:
        from app import app
        
        logger.info(f"Starting Flask application on port {args.port}")
        app.run(debug=args.debug, host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Error starting Flask application: {e}")

def analyze_pcap(args):
    """Analyze a single PCAP file using the loaded model."""
    from utils.pcap_utils import predict_from_pcap
    import pandas as pd
    import json
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    if 'xgboost' in args.model.lower():
        from models.xgboost_model import XGBoostModel
        model = XGBoostModel(model_path=args.model)
    elif 'deep' in args.model.lower():
        from models.deep_model import DeepModel
        model = DeepModel(model_path=args.model)
    else:
        from models.ensemble import EnsembleModel
        model = EnsembleModel(model_path=args.model)
    
    # Load scaler if provided
    scaler = None
    if args.scaler:
        import joblib
        scaler = joblib.load(args.scaler)
        logger.info(f"Loaded scaler from {args.scaler}")
    
    # Analyze PCAP file
    logger.info(f"Analyzing PCAP file: {args.file}")
    results_df = predict_from_pcap(args.file, model, scaler)
    
    if results_df.empty:
        logger.warning("No flows extracted from PCAP file")
        return
    
    # Count predictions by class
    prediction_counts = results_df['prediction'].value_counts().to_dict()
    
    # Map class indices to class names if available
    if hasattr(model, 'class_names') and model.class_names:
        prediction_counts = {
            model.class_names[int(k)] if isinstance(k, (int, float)) else k: v 
            for k, v in prediction_counts.items()
        }
    
    # Calculate statistics
    total_flows = len(results_df)
    benign_count = prediction_counts.get('benign', prediction_counts.get(0, 0))
    malicious_count = total_flows - benign_count
    malicious_percent = (malicious_count / total_flows) * 100 if total_flows > 0 else 0
    
    # Display results
    logger.info(f"Analysis Results for {args.file}:")
    logger.info(f"Total flows: {total_flows}")
    logger.info(f"Benign flows: {benign_count}")
    logger.info(f"Malicious flows: {malicious_count} ({malicious_percent:.2f}%)")
    logger.info(f"Predictions by class: {json.dumps(prediction_counts, indent=2)}")
    
    # Save results if output file is provided
    if args.output:
        results_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
    
    return results_df

def main():
    """Main entry point for the NIDS application."""
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup environment parser
    setup_parser = subparsers.add_parser('setup', help='Set up the environment')
    
    # Convert parser
    convert_parser = subparsers.add_parser('convert', help='Convert parquet files to CSV')
    convert_parser.add_argument('--input', '-i', type=str, required=True,
                               help='Input directory containing parquet files')
    convert_parser.add_argument('--output', '-o', type=str, default=None,
                               help='Output directory for CSV files')
    convert_parser.add_argument('--recursive', '-r', action='store_true',
                               help='Search recursively for parquet files')
    convert_parser.add_argument('--merge', '-m', type=str, default=None,
                               help='Merge CSV files into a single file')
    convert_parser.add_argument('--sample-size', '-s', type=int, default=None,
                               help='Sample size from each file when merging')
    
    # Preprocess parser
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('--pcap-dir', type=str, default=None,
                                 help='Directory containing PCAP files')
    preprocess_parser.add_argument('--output-dir', type=str, default=None,
                                 help='Directory for output files')
    preprocess_parser.add_argument('--recursive', action='store_true',
                                 help='Search recursively for PCAP files')
    preprocess_parser.add_argument('--input-csv', type=str, default=None,
                                 help='Path to input CSV file with flow features')
    preprocess_parser.add_argument('--label-file', type=str, default=None,
                                 help='Path to CSV file with labels')
    preprocess_parser.add_argument('--combine', type=str, nargs='+', default=None,
                                 help='List of CSV files to combine')
    preprocess_parser.add_argument('--sample-size', type=int, default=None,
                                 help='Number of samples to take from each file when combining')
    
    # Train parser
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data', '-d', type=str, required=True,
                            help='Path to dataset CSV file')
    train_parser.add_argument('--label-column', '-l', type=str, default='Label',
                            help='Name of the label column')
    train_parser.add_argument('--model-type', '-m', type=str, default='ensemble',
                            choices=['xgboost', 'deep', 'ensemble'],
                            help='Type of model to train')
    train_parser.add_argument('--output-dir', '-o', type=str, default=None,
                            help='Directory to save trained models')
    train_parser.add_argument('--test', '-t', action='store_true',
                            help='Run in test mode with reduced dataset')
    
    # Run parser
    run_parser = subparsers.add_parser('run', help='Run the Flask application')
    run_parser.add_argument('--host', type=str, default='0.0.0.0',
                          help='Host to run the application on')
    run_parser.add_argument('--port', type=int, default=5000,
                          help='Port to run the application on')
    run_parser.add_argument('--debug', action='store_true',
                          help='Run in debug mode')
    
    # Analyze parser
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a PCAP file')
    analyze_parser.add_argument('--file', '-f', type=str, required=True,
                              help='Path to PCAP file')
    analyze_parser.add_argument('--model', '-m', type=str, required=True,
                              help='Path to model file or directory')
    analyze_parser.add_argument('--scaler', '-s', type=str, default=None,
                              help='Path to scaler file')
    analyze_parser.add_argument('--output', '-o', type=str, default=None,
                              help='Path to output CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run appropriate function based on command
    if args.command == 'setup':
        setup_environment()
    elif args.command == 'convert':
        convert_parquet_files(args)
    elif args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'run':
        run_flask_app(args)
    elif args.command == 'analyze':
        analyze_pcap(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    logger.info(f"Completed in {elapsed_time:.2f} seconds")