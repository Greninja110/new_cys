"""
Flask application for the NIDS system.
"""
import os
import sys
import pandas as pd
import numpy as np
import time
import joblib
import json
import uuid
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
import plotly
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import get_logger
from utils.pcap_utils import pcap_to_csv, predict_from_pcap
from models.ensemble import EnsembleModel
from config import FLASK_SECRET_KEY, UPLOAD_FOLDER, MODEL_DIR, ALLOWED_EXTENSIONS

# Initialize logger
logger = get_logger('app')

# Create Flask app
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pretrained model
model = None
model_info = None
scaler = None

def load_model(model_path=None):
    """
    Load the pretrained model.
    
    Args:
        model_path (str, optional): Path to model directory
        
    Returns:
        tuple: model, model_info, scaler
    """
    global model, model_info, scaler
    
    # Find the latest model if path not provided
    if model_path is None:
        model_dirs = [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) 
                     if os.path.isdir(os.path.join(MODEL_DIR, d))]
        if not model_dirs:
            logger.error("No model directories found")
            return None, None, None
        
        # Get the most recent model
        model_path = max(model_dirs, key=os.path.getmtime)
    
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Determine model type based on directory name
        model_type = os.path.basename(model_path).split('_')[0]
        
        # Load scaler
        scaler_path = os.path.join(model_path, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            scaler = None
            logger.warning("No scaler found")
        
        # Load model
        if model_type == 'xgboost':
            from models.xgboost_model import XGBoostModel
            model_file = os.path.join(model_path, 'xgboost_model.json')
            model = XGBoostModel(model_path=model_file)
        elif model_type == 'deep':
            from models.deep_model import DeepModel
            model_file = os.path.join(model_path, 'deep_model.h5')
            model = DeepModel(model_path=model_file)
        else:  # ensemble
            model_file = os.path.join(model_path, 'ensemble_model')
            model = EnsembleModel(model_path=model_file)
        
        # Get model information
        model_info = {
            'type': model_type,
            'path': model_path,
            'timestamp': datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S'),
            'class_names': model.class_names
        }
        
        logger.info(f"Model loaded successfully: {model_info}")
        
        return model, model_info, scaler
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None, None, None

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    global model_info
    return render_template('index.html', model_info=model_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction."""
    global model, scaler
    
    # Check if model is loaded
    if model is None:
        model, model_info, scaler = load_model()
        if model is None:
            flash('No model available. Please train a model first.', 'error')
            return redirect(url_for('index'))
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # If user does not select file
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{unique_id}_{filename}"
        
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Process file and make predictions
        try:
            # Convert PCAP to CSV
            csv_file = pcap_to_csv(file_path)
            
            # Make predictions
            results_df = predict_from_pcap(file_path, model, scaler)
            
            if results_df.empty:
                flash('No network flows extracted from PCAP file', 'warning')
                return redirect(url_for('index'))
            
            # Save predictions
            results_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(unique_filename)[0]}_results.csv")
            results_df.to_csv(results_file, index=False)
            
            # Count predictions by class
            prediction_counts = results_df['prediction'].value_counts().to_dict()
            
            # Map class indices to class names if available
            if hasattr(model, 'class_names') and model.class_names:
                prediction_counts = {model.class_names[int(k)] if isinstance(k, (int, np.integer)) else k: v 
                                    for k, v in prediction_counts.items()}
            
            # Calculate statistics
            total_flows = len(results_df)
            benign_count = prediction_counts.get('benign', prediction_counts.get(0, 0))
            malicious_count = total_flows - benign_count
            malicious_percent = (malicious_count / total_flows) * 100 if total_flows > 0 else 0
            
            # Calculate top source and destination IPs
            top_src_ips = results_df['src_ip'].value_counts().head(5).to_dict()
            top_dst_ips = results_df['dst_ip'].value_counts().head(5).to_dict()
            
            # Create visualizations with plotly
            # Prediction distribution pie chart
            fig1 = px.pie(
                values=list(prediction_counts.values()),
                names=list(prediction_counts.keys()),
                title='Traffic Classification'
            )
            chart1_json = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Traffic over time if timestamp column exists
            chart2_json = None
            if 'timestamp' in results_df.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(results_df['timestamp']):
                    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
                
                # Group by time and prediction
                results_df['hour'] = results_df['timestamp'].dt.floor('H')
                time_series = results_df.groupby(['hour', 'prediction']).size().reset_index(name='count')
                
                fig2 = px.line(
                    time_series, 
                    x='hour', 
                    y='count', 
                    color='prediction',
                    title='Traffic Over Time'
                )
                chart2_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Protocol distribution
            if 'protocol' in results_df.columns:
                protocol_counts = results_df['protocol'].value_counts().head(10)
                
                # Map protocol numbers to names
                protocol_map = {
                    1: 'ICMP',
                    6: 'TCP',
                    17: 'UDP',
                    47: 'GRE',
                    50: 'ESP',
                    51: 'AH',
                    58: 'ICMPv6',
                    132: 'SCTP'
                }
                
                protocol_counts.index = [protocol_map.get(p, f"Protocol {p}") for p in protocol_counts.index]
                
                fig3 = px.bar(
                    x=protocol_counts.index,
                    y=protocol_counts.values,
                    title='Protocol Distribution'
                )
                chart3_json = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
            else:
                chart3_json = None
            
            # Prepare result data
            result_data = {
                'filename': file.filename,
                'upload_time': timestamp,
                'total_flows': total_flows,
                'benign_count': benign_count,
                'malicious_count': malicious_count,
                'malicious_percent': malicious_percent,
                'prediction_counts': prediction_counts,
                'top_src_ips': top_src_ips,
                'top_dst_ips': top_dst_ips,
                'chart1_json': chart1_json,
                'chart2_json': chart2_json,
                'chart3_json': chart3_json,
                'results_file': os.path.basename(results_file)
            }
            
            return render_template('results.html', result=result_data, model_info=model_info)
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Only PCAP files are allowed.', 'error')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file from the upload folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/dashboard')
def dashboard():
    """Render the dashboard page."""
    # List recent analysis results
    results_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('_results.csv')]
    results_files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)
    
    # Limit to 10 most recent
    results_files = results_files[:10]
    
    # Load data for each result file
    results_data = []
    
    for result_file in results_files:
        try:
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], result_file))
            
            # Extract original filename from result filename
            filename = '_'.join(result_file.split('_')[2:-1])
            
            # Get result statistics
            total_flows = len(df)
            
            malicious_count = 0
            if 'prediction' in df.columns:
                # Count non-benign predictions
                if df['prediction'].dtype == 'object':
                    malicious_count = len(df[df['prediction'] != 'benign'])
                else:
                    malicious_count = len(df[df['prediction'] != 0])
            
            timestamp = ' '.join(result_file.split('_')[:2])
            
            results_data.append({
                'filename': filename,
                'timestamp': timestamp,
                'total_flows': total_flows,
                'malicious_count': malicious_count,
                'result_file': result_file
            })
            
        except Exception as e:
            logger.warning(f"Error reading result file {result_file}: {str(e)}")
            continue
    
    return render_template('dashboard.html', results=results_data, model_info=model_info)

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html', model_info=model_info)

@app.route('/models')
def models_page():
    """Render the models page."""
    # List available models
    model_dirs = [d for d in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
    
    # Get information about each model
    models_data = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(MODEL_DIR, model_dir)
        
        # Get model type
        model_type = model_dir.split('_')[0]
        
        # Get timestamp
        try:
            timestamp = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
        except:
            timestamp = 'Unknown'
        
        # Check for evaluation metrics
        eval_dir = os.path.join(model_path, 'evaluation')
        has_evaluation = os.path.exists(eval_dir)
        
        models_data.append({
            'name': model_dir,
            'type': model_type,
            'timestamp': timestamp,
            'path': model_path,
            'has_evaluation': has_evaluation
        })
    
    # Sort by timestamp (newest first)
    models_data.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('models.html', models=models_data, model_info=model_info)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """Load a specific model."""
    model_path = request.form.get('model_path')
    
    if not model_path:
        flash('No model path provided', 'error')
        return redirect(url_for('models_page'))
    
    global model, model_info, scaler
    model, model_info, scaler = load_model(model_path)
    
    if model is None:
        flash('Error loading model', 'error')
    else:
        flash(f'Model {os.path.basename(model_path)} loaded successfully', 'success')
    
    return redirect(url_for('models_page'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    global model, scaler
    
    # Check if model is loaded
    if model is None:
        model, model_info, scaler = load_model()
        if model is None:
            return jsonify({'error': 'No model available'}), 500
    
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{unique_id}_{filename}"
        
        # Save file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Make predictions
            results_df = predict_from_pcap(file_path, model, scaler)
            
            if results_df.empty:
                return jsonify({'error': 'No network flows extracted from PCAP file'}), 400
            
            # Count predictions by class
            prediction_counts = results_df['prediction'].value_counts().to_dict()
            
            # Map class indices to class names if available
            if hasattr(model, 'class_names') and model.class_names:
                prediction_counts = {model.class_names[int(k)] if isinstance(k, (int, np.integer)) else k: v 
                                    for k, v in prediction_counts.items()}
            
            # Calculate statistics
            total_flows = len(results_df)
            benign_count = prediction_counts.get('benign', prediction_counts.get(0, 0))
            malicious_count = total_flows - benign_count
            malicious_percent = (malicious_count / total_flows) * 100 if total_flows > 0 else 0
            
            # Prepare response
            response = {
                'filename': file.filename,
                'total_flows': total_flows,
                'benign_count': benign_count,
                'malicious_count': malicious_count,
                'malicious_percent': malicious_percent,
                'prediction_counts': prediction_counts
            }
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type. Only PCAP files are allowed.'}), 400

if __name__ == '__main__':
    # Load model
    model, model_info, scaler = load_model()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)