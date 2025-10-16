# src/config.py
# Configuration management for inference pipeline

from pathlib import Path
from typing import Dict, Any
import os

class Config:
    """
    Central configuration management.
    
    Manages all paths, hyperparameters, and settings for the inference pipeline.
    Provides validation and summary methods for operational transparency.
    """
    
    def __init__(self, base_dir: str = None):
        """
        Initialize configuration with default values.
        
        Args:
            base_dir: Base directory path (defaults to parent of src/)
        """
        # Base paths
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        # Data paths - adjusted for evaluation server structure
        self.paths = {
            'data_dir': self.base_dir / 'data',
            'test': self.base_dir / 'data' / 'test.csv',
            'test_a': self.base_dir / 'data' / 'test' / 'A.csv',
            'test_b': self.base_dir / 'data' / 'test' / 'B.csv',
            'model_dir': self.base_dir / 'model',
            'model': self.base_dir / 'model' / 'lgbm_A.pkl',
            'model_b': self.base_dir / 'model' / 'lgbm_B.pkl',
            'output_dir': self.base_dir / 'output',
            'output': self.base_dir / 'output' / 'submission.csv',
            'log_dir': self.base_dir / 'output',
            'result_log': self.base_dir / 'output' / 'experiment_results.txt'
        }
        
        # Inference parameters
        self.inference = {
            'batch_size': 10000,
            'num_threads': 3,
            'use_progress': False
        }
        
        # Preprocessing settings
        self.preprocessing = {
            'handle_missing': 'median',
            'eps': 1e-6
        }
        
        # Logging
        self.logging = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            FileNotFoundError: If required files do not exist
            ValueError: If configuration values are invalid
        """
        # Check test file exists
        if not self.paths['test'].exists():
            raise FileNotFoundError(f"Test file not found: {self.paths['test']}")
        
        # Check model directory exists
        if not self.paths['model_dir'].exists():
            raise FileNotFoundError(f"Model directory not found: {self.paths['model_dir']}")
        
        # Validate inference parameters
        if self.inference['batch_size'] <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.inference['num_threads'] <= 0:
            raise ValueError("num_threads must be positive")
        
        return True
    
    def get_summary(self) -> str:
        """
        Get configuration summary string.
        
        Returns:
            str: Configuration summary
        """
        return (f"batch_size={self.inference['batch_size']}, "
                f"num_threads={self.inference['num_threads']}")
    
    def ensure_output_dir(self) -> None:
        """Create output directory if it does not exist."""
        self.paths['output_dir'].mkdir(parents=True, exist_ok=True)
    
    def ensure_log_dir(self) -> None:
        """Create log directory if it does not exist."""
        self.paths['log_dir'].mkdir(parents=True, exist_ok=True)