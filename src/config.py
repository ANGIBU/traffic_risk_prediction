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
            'feature_names_a': self.base_dir / 'model' / 'feature_names_A.txt',
            'feature_names_b': self.base_dir / 'model' / 'feature_names_B.txt',
            'output_dir': self.base_dir / 'output',
            'output': self.base_dir / 'output' / 'submission.csv',
            'log_dir': self.base_dir / 'output',
            'result_log': self.base_dir / 'output' / 'experiment_results.txt'
        }
        
        # Inference parameters
        self.inference = {
            'batch_size': 15000,
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
            'use_nonlinear': True,
            'remove_correlated': True,
            'correlation_threshold': 0.90,
            'use_domain_features': True,
            'use_log_transform': True
        }
        
        # Training settings
        self.training = {
            'n_splits': 5,
            'random_state': 42,
            'stratified': True,
            'verbose_eval': 50,
            'early_stopping_rounds': 50,
            'early_stopping_rounds_b': 75,
            'use_feature_selection': True,
            'feature_selection_threshold': 0.85,
            'feature_selection_threshold_b': 0.78,
            'use_calibration': True,
            'remove_correlated_features': True,
            'use_ensemble': True,
            'ensemble_top_k': 3,
            'use_smote_b': True,
            'smote_sampling_strategy': 0.30,
            'use_oof_calibration': True
        }
        
        # LightGBM hyperparameters for Type A
        self.lgbm_params_a = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'min_child_samples': 35,
            'min_child_weight': 0.001,
            'learning_rate': 0.03,
            'n_estimators': 2000,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10.0,
            'is_unbalance': False,
            'reg_alpha': 1.2,
            'reg_lambda': 1.2,
            'min_split_gain': 0.02,
            'n_jobs': 3,
            'verbose': -1,
            'random_state': 42,
            'importance_type': 'gain'
        }
        
        # LightGBM hyperparameters for Type B
        self.lgbm_params_b = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 7,
            'min_child_samples': 35,
            'min_child_weight': 0.001,
            'learning_rate': 0.025,
            'n_estimators': 2000,
            'subsample': 0.85,
            'subsample_freq': 1,
            'colsample_bytree': 0.9,
            'scale_pos_weight': 12.0,
            'is_unbalance': False,
            'reg_alpha': 0.8,
            'reg_lambda': 0.8,
            'min_split_gain': 0.01,
            'n_jobs': 3,
            'verbose': -1,
            'random_state': 42,
            'importance_type': 'gain'
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