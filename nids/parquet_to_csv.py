"""
Convert parquet files to CSV format.
"""
import os
import sys
import glob
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import time
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.logger import get_logger
from config import DATA_DIR

# Initialize logger
logger = get_logger('parquet_to_csv')

def convert_parquet_to_csv(parquet_file, output_file=None, show_preview=True):
    """
    Convert a single parquet file to CSV.
    
    Args:
        parquet_file (str): Path to parquet file
        output_file (str, optional): Path to output CSV file
        show_preview (bool): Whether to show a preview of the data
        
    Returns:
        str: Path to output CSV file
    """
    start_time = time.time()
    logger.info(f"Converting {parquet_file} to CSV")
    
    try:
        # Generate output filename if not provided
        if output_file is None:
            output_file = os.path.splitext(parquet_file)[0] + '.csv'
        
        # Read parquet file
        df = pq.read_table(parquet_file).to_pandas()
        
        # Log basic info about the dataframe
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Memory usage: {df.memory_usage().sum() / (1024**2):.2f} MB")
        
        # Show preview if requested
        if show_preview:
            logger.info(f"Preview of the data:")
            logger.info("\nColumns:")
            logger.info(", ".join(df.columns.tolist()))
            logger.info("\nData types:")
            for col, dtype in df.dtypes.items():
                logger.info(f"{col}: {dtype}")
            logger.info("\nSample data (first 5 rows):")
            logger.info(df.head().to_string())
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                logger.info("\nMissing values by column:")
                for col, count in missing_values[missing_values > 0].items():
                    logger.info(f"{col}: {count} ({count/len(df)*100:.2f}%)")
        
        # Write to CSV
        df.to_csv(output_file, index=False)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Conversion completed in {elapsed_time:.2f} seconds")
        logger.info(f"CSV file saved to {output_file}")
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error converting parquet to CSV: {e}")
        raise

def convert_all_parquet_files(input_dir, output_dir=None, recursive=True):
    """
    Convert all parquet files in a directory to CSV.
    
    Args:
        input_dir (str): Directory containing parquet files
        output_dir (str, optional): Directory for output CSV files
        recursive (bool): Whether to search recursively
        
    Returns:
        list: Paths to output CSV files
    """
    logger.info(f"Converting all parquet files in {input_dir}")
    
    # Set output directory
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files
    search_pattern = os.path.join(input_dir, '**/*.parquet') if recursive else os.path.join(input_dir, '*.parquet')
    parquet_files = glob.glob(search_pattern, recursive=recursive)
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {input_dir}")
        return []
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Convert each file
    output_files = []
    
    for parquet_file in tqdm(parquet_files, desc="Converting files"):
        # Generate output file path
        rel_path = os.path.relpath(parquet_file, input_dir)
        output_file = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.csv')
        
        # Create output subdirectory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            # Convert file
            converted_file = convert_parquet_to_csv(parquet_file, output_file, show_preview=(len(parquet_files) < 5))
            output_files.append(converted_file)
        
        except Exception as e:
            logger.error(f"Error converting {parquet_file}: {e}")
            continue
    
    logger.info(f"Converted {len(output_files)} out of {len(parquet_files)} files")
    
    return output_files

def merge_csv_files(input_files, output_file, sample_size=None):
    """
    Merge multiple CSV files into a single file.
    
    Args:
        input_files (list): List of CSV files to merge
        output_file (str): Path to output merged CSV file
        sample_size (int, optional): Sample size from each file (to handle large files)
        
    Returns:
        str: Path to output merged CSV file
    """
    logger.info(f"Merging {len(input_files)} CSV files")
    
    dfs = []
    
    for csv_file in tqdm(input_files, desc="Reading files"):
        try:
            # Read file, with sampling if requested
            if sample_size is not None:
                # Get total number of rows
                total_rows = sum(1 for _ in open(csv_file)) - 1  # subtract header
                
                if total_rows > sample_size:
                    # Calculate sampling fraction
                    fraction = sample_size / total_rows
                    df = pd.read_csv(csv_file, skiprows=lambda x: x > 0 and np.random.rand() > fraction)
                else:
                    df = pd.read_csv(csv_file)
            else:
                df = pd.read_csv(csv_file)
            
            dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
            continue
    
    if not dfs:
        logger.error("No data to merge")
        return None
    
    # Merge dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged dataframe
    merged_df.to_csv(output_file, index=False)
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    logger.info(f"Merged file saved to {output_file}")
    
    return output_file

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert parquet files to CSV')
    parser.add_argument('--input', '-i', type=str, default=DATA_DIR,
                        help='Input directory containing parquet files')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for CSV files')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Search recursively for parquet files')
    parser.add_argument('--merge', '-m', type=str, default=None,
                        help='Merge CSV files into a single file')
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Sample size from each file when merging')
    
    args = parser.parse_args()
    
    # Convert files
    csv_files = convert_all_parquet_files(args.input, args.output, args.recursive)
    
    # Merge if requested
    if args.merge and csv_files:
        merge_csv_files(csv_files, args.merge, args.sample)