# src/config.py
# Configuration management for inference and training pipeline

from pathlib import Path
from typing import Dict, Any
import os

class Config:
    """
    Central configuration management for both inference and training.
    
    Manages paths, hyperparameters, and settings with validation.
    """
    
    def __init__(self, base_dir: str = None, mode: str = 'inference'):
        """
        Initialize configuration.
        
        Args:
            base_dir: Base directory path (defaults to parent of src/)
            mode: 'inference' or 'training'
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.mode = mode
        
        # Data paths
        self.paths = {
            'data_dir': self.base_dir / 'data',
            'test': self.base_dir / 'data' / 'test.csv',
            'test_a': self.base_dir / 'data' / 'test' / 'A.csv',
            'test_b': self.base_dir / 'data' / 'test' / 'B.csv',
            'data_dev_dir': self.base_dir / 'data_dev',
            'train': self.base_dir / 'data_dev' / 'train.csv',
            'train_a': self.base_dir / 'data_dev' / 'train' / 'A.csv',
            'train_b': self.base_dir / 'data_dev' / 'train' / 'B.csv',
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
        
        # Feature engineering settings
        self.feature_engineering = {
            'use_temporal': True,
            'use_cross_test': True,
            'use_interaction': True,
            'use_nonlinear': True
        }
        
        # Training settings
        self.training = {
            'n_splits': 5,
            'random_state': 42,
            'stratified': True,
            'verbose_eval': 50,
            'early_stopping_rounds': 100
        }
        
        # LightGBM hyperparameters for Type A
        self.lgbm_params_a = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 10,
            'learning_rate': 0.03,
            'n_estimators': 500,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 33.6,
            'is_unbalance': True,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': 3,
            'verbose': -1,
            'random_state': 42
        }
        
        # LightGBM hyperparameters for Type B
        self.lgbm_params_b = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 8,
            'min_child_samples': 10,
            'learning_rate': 0.03,
            'n_estimators': 500,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 33.6,
            'is_unbalance': True,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_jobs': 3,
            'verbose': -1,
            'random_state': 42
        }
        
        # Logging
        self.logging = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        
    def validate(self) -> bool:
        """
        Validate configuration settings based on mode.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            FileNotFoundError: If required files do not exist
            ValueError: If configuration values are invalid
        """
        if self.mode == 'inference':
            if not self.paths['test'].exists():
                raise FileNotFoundError(f"Test file not found: {self.paths['test']}")
        elif self.mode == 'training':
            if not self.paths['train'].exists():
                raise FileNotFoundError(f"Train file not found: {self.paths['train']}")
        
        if not self.paths['model_dir'].exists():
            self.paths['model_dir'].mkdir(parents=True, exist_ok=True)
        
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
    
    def ensure_model_dir(self) -> None:
        """Create model directory if it does not exist."""
        self.paths['model_dir'].mkdir(parents=True, exist_ok=True)