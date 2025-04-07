"""
Logger utility for the NIDS project.
Handles logging to both console and file.
"""
import os
import sys
import logging
from datetime import datetime
import traceback
from logging.handlers import RotatingFileHandler

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_DIR, LOG_FILE

class Logger:
    """
    Custom logger class for the NIDS project.
    """
    def __init__(self, name, log_file=LOG_FILE, level=logging.INFO):
        """
        Initialize logger with name, file and level.
        
        Args:
            name (str): Logger name
            log_file (str): Path to log file
            level (int): Logging level
        """
        # Create logs directory if it doesn't exist
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Create logger instance
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear existing handlers if any
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] - %(name)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] - %(name)s - %(message)s'
        )
        
        # File handler with rotation (10MB max, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Logger initialized: {name}")
    
    def get_logger(self):
        """Return the logger instance."""
        return self.logger
    
    def exception_handler(self, exc_type, exc_value, exc_traceback):
        """
        Custom exception handler to log exceptions.
        
        Args:
            exc_type: Exception type
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # Skip KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Log the exception
        self.logger.error("Uncaught exception:", 
                         exc_info=(exc_type, exc_value, exc_traceback))
        
        # Print traceback to console
        traceback.print_exception(exc_type, exc_value, exc_traceback)

# Create a default logger instance
default_logger = Logger('nids').get_logger()

# Set the default exception handler
sys.excepthook = Logger('nids').exception_handler

# Function to get a new logger for a specific module
def get_logger(name):
    """
    Get a configured logger for a specific module.
    
    Args:
        name (str): Name of the module
        
    Returns:
        logging.Logger: Configured logger
    """
    return Logger(name).get_logger()