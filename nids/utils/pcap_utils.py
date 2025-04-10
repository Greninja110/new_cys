"""
Utilities for handling PCAP files.
"""
import os
import sys
import pandas as pd
import numpy as np
import dpkt
import socket
import struct
from collections import defaultdict
import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from config import PCAP_FEATURES

# Initialize logger
logger = get_logger('pcap_utils')

def mac_addr(address):
    """Convert MAC address to string."""
    return ':'.join('%02x' % b for b in address)

def inet_to_str(inet):
    """Convert inet object to string."""
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

def extract_features_from_pcap(pcap_file):
    """
    Extract network traffic features from PCAP file.
    
    Args:
        pcap_file (str): Path to PCAP file
        
    Returns:
        pandas.DataFrame: DataFrame with extracted features
    """
    logger.info(f"Extracting features from {pcap_file}")
    
    # Initialize data structures
    packets = []
    flows = defaultdict(list)
    
    try:
        # Open the pcap file
        with open(pcap_file, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            
            # Process each packet
            for ts, buf in tqdm(pcap, desc="Processing packets"):
                try:
                    # Parse ethernet frame
                    eth = dpkt.ethernet.Ethernet(buf)
                    
                    # Skip non-IP packets
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    
                    ip = eth.data
                    
                    # Get source and destination IP
                    src_ip = inet_to_str(ip.src)
                    dst_ip = inet_to_str(ip.dst)
                    
                    # Get protocol
                    protocol = ip.p
                    
                    # Default values
                    src_port = dst_port = 0
                    tcp_flags = tcp_window_size = 0
                    
                    # Extract TCP/UDP specific info
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        tcp = ip.data
                        src_port = tcp.sport
                        dst_port = tcp.dport
                        tcp_flags = tcp.flags
                        tcp_window_size = tcp.win
                        payload = tcp.data
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        udp = ip.data
                        src_port = udp.sport
                        dst_port = udp.dport
                        payload = udp.data
                    else:
                        payload = b''
                    
                    # Create flow key (5-tuple)
                    forward_flow = (src_ip, dst_ip, protocol, src_port, dst_port)
                    reverse_flow = (dst_ip, src_ip, protocol, dst_port, src_port)
                    
                    # Store packet info
                    packet_info = {
                        'timestamp': datetime.datetime.utcfromtimestamp(ts),
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'protocol': protocol,
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'packet_length': len(buf),
                        'tcp_flags': tcp_flags,
                        'tcp_window_size': tcp_window_size,
                        'ttl': ip.ttl,
                        'payload_length': len(payload)
                    }
                    
                    # Add to packets list
                    packets.append(packet_info)
                    
                    # Add to flows
                    if forward_flow in flows or reverse_flow not in flows:
                        flows[forward_flow].append((ts, packet_info))
                    else:
                        flows[reverse_flow].append((ts, packet_info))
                        
                except Exception as e:
                    logger.debug(f"Error processing packet: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error processing PCAP file: {e}")
        raise
    
    # Create packets DataFrame
    df_packets = pd.DataFrame(packets)
    
    # Create flows DataFrame with aggregated features
    flow_features = []
    
    for flow_key, packets_info in flows.items():
        if not packets_info:
            continue
            
        # Sort packets by timestamp
        packets_info.sort(key=lambda x: x[0])
        
        # Extract timestamps and packet info
        timestamps = [p[0] for p in packets_info]
        packet_infos = [p[1] for p in packets_info]
        
        # Calculate flow duration
        flow_duration = max(timestamps) - min(timestamps)
        
        # Count packets
        packet_count = len(packets_info)
        
        # Get first packet info
        first_packet = packet_infos[0]
        
        # Calculate flow features
        flow_feature = {
            'src_ip': first_packet['src_ip'],
            'dst_ip': first_packet['dst_ip'],
            'protocol': first_packet['protocol'],
            'src_port': first_packet['src_port'],
            'dst_port': first_packet['dst_port'],
            'packet_count': packet_count,
            'flow_duration': flow_duration,
            'timestamp': datetime.datetime.utcfromtimestamp(min(timestamps)),
            'packet_length_mean': np.mean([p['packet_length'] for p in packet_infos]),
            'packet_length_std': np.std([p['packet_length'] for p in packet_infos]),
            'packet_length_min': min([p['packet_length'] for p in packet_infos]),
            'packet_length_max': max([p['packet_length'] for p in packet_infos]),
            'payload_length_mean': np.mean([p['payload_length'] for p in packet_infos]),
            'payload_length_total': sum([p['payload_length'] for p in packet_infos]),
            'ttl_mean': np.mean([p['ttl'] for p in packet_infos]),
        }
        
        # Get TCP specific features if applicable
        if first_packet['protocol'] == 6:  # TCP
            tcp_flags_all = [p['tcp_flags'] for p in packet_infos]
            tcp_window_sizes = [p['tcp_window_size'] for p in packet_infos]
            
            # Calculate TCP specific features
            flow_feature.update({
                'tcp_flags_sum': sum(tcp_flags_all),
                'tcp_window_size_mean': np.mean(tcp_window_sizes),
                'tcp_window_size_std': np.std(tcp_window_sizes),
            })
        
        flow_features.append(flow_feature)
    
    # Create flows DataFrame
    df_flows = pd.DataFrame(flow_features)
    
    logger.info(f"Extracted {len(df_packets)} packets and {len(df_flows)} flows")
    
    return df_packets, df_flows

def pcap_to_csv(pcap_file, output_file=None):
    """
    Convert PCAP file to CSV.
    
    Args:
        pcap_file (str): Path to PCAP file
        output_file (str, optional): Path to output CSV file
        
    Returns:
        str: Path to the output CSV file
    """
    logger.info(f"Converting {pcap_file} to CSV")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = os.path.splitext(pcap_file)[0] + '.csv'
    
    try:
        # Extract features
        df_packets, df_flows = extract_features_from_pcap(pcap_file)
        
        # Save to CSV
        df_flows.to_csv(output_file, index=False)
        logger.info(f"Saved CSV to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error converting PCAP to CSV: {e}")
        raise

def predict_from_pcap(pcap_file, model, scaler=None):
    """
    Make predictions directly from a PCAP file.
    
    Args:
        pcap_file (str): Path to PCAP file
        model: Trained model for prediction
        scaler: Fitted scaler for feature normalization
        
    Returns:
        pandas.DataFrame: DataFrame with predictions
    """
    logger.info(f"Making predictions from {pcap_file}")
    
    try:
        # Extract features
        _, df_flows = extract_features_from_pcap(pcap_file)
        
        if df_flows.empty:
            logger.warning("No flows extracted from PCAP file")
            return pd.DataFrame()
        
        # Create a copy of the original dataframe
        original_df = df_flows.copy()
        
        # Initial feature mapping (lowercase snake_case to CamelCase)
        feature_map = {
            'src_ip': 'Src IP',
            'dst_ip': 'Dst IP',
            'protocol': 'Protocol',
            'src_port': 'Src Port',
            'dst_port': 'Dst Port',
            'flow_duration': 'Flow Duration',
            'packet_count': 'Total Fwd Packets',  # Approximation
            'packet_length_mean': 'Packet Length Mean',
            'packet_length_std': 'Packet Length Std',
            'packet_length_min': 'Packet Length Min',
            'packet_length_max': 'Packet Length Max',
            'ttl_mean': 'TTL Mean',
            'tcp_flags_sum': 'SYN Flag Count',  # Approximation 
            'tcp_window_size_mean': 'Init Fwd Win Bytes',  # Approximation
            'payload_length_mean': 'Fwd Packets Length Total',  # Approximation
            'payload_length_total': 'Bwd Packets Length Total',  # Approximation
        }
        
        # Create a new DataFrame with the proper structure for the model
        X = pd.DataFrame()
        
        # Get the expected feature names from the model
        if hasattr(model, 'feature_names') and model.feature_names:
            expected_features = model.feature_names
        elif hasattr(model, 'xgb_model') and hasattr(model.xgb_model, 'feature_names'):
            expected_features = model.xgb_model.feature_names
        else:
            logger.warning("Model doesn't have feature_names attribute. Using scaler feature names.")
            expected_features = scaler.feature_names_in_
        
        logger.info(f"Expected features from model: {expected_features[:5]}...")
        logger.info(f"Features from PCAP: {df_flows.columns[:5]}...")
        
        # Convert all extracted features to lowercase for easier mapping
        extracted_cols = {col.lower().replace(' ', '_'): col for col in df_flows.columns}
        
        # Create a dummy dataframe with all expected features initialized to 0
        X = pd.DataFrame(0, index=range(len(df_flows)), columns=expected_features)
        
        # Try to map PCAP features to expected features
        for model_feat in expected_features:
            # Try direct mapping (case insensitive)
            pcap_feat = model_feat.lower().replace(' ', '_')
            if pcap_feat in extracted_cols:
                X[model_feat] = df_flows[extracted_cols[pcap_feat]]
                logger.debug(f"Mapped {extracted_cols[pcap_feat]} to {model_feat}")
            # Try through the feature map
            else:
                mapped = False
                for pcap_key, model_key in feature_map.items():
                    if model_key == model_feat and pcap_key in df_flows.columns:
                        X[model_feat] = df_flows[pcap_key]
                        mapped = True
                        logger.debug(f"Mapped {pcap_key} to {model_feat} via map")
                        break
                
                if not mapped:
                    logger.debug(f"No mapping found for {model_feat}, using default value 0")
        
        # Handle missing values
        X.fillna(0, inplace=True)
        
        logger.info(f"Prepared feature matrix with shape: {X.shape}")
        
        # Apply scaling if provided
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        
        # Add predictions to the original DataFrame
        original_df['prediction'] = y_pred
        
        return original_df
    
    except Exception as e:
        logger.error(f"Error making predictions from PCAP: {e}", exc_info=True)
        raise