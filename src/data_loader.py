# src/data_loader.py
# Data loading utilities with memory optimization

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loading and initial validation.
    
    Handles loading test data from CSV files with memory optimization
    and schema validation.
    """
    
    def __init__(self, config):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.paths = config.paths
        
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load test data with A/B test details.
        
        Returns:
            tuple: (test_meta, test_a, test_b) DataFrames
        """
        # Load main test file
        logger.info(f"Loading test metadata from {self.paths['test']}")
        test_meta = pd.read_csv(self.paths['test'])
        logger.info(f"Loaded test metadata: {test_meta.shape}")
        
        # Load test A and B details
        logger.info(f"Loading test A from {self.paths['test_a']}")
        test_a = self._load_with_optimization(self.paths['test_a'])
        logger.info(f"Loaded test A: {test_a.shape}")
        
        logger.info(f"Loading test B from {self.paths['test_b']}")
        test_b = self._load_with_optimization(self.paths['test_b'])
        logger.info(f"Loaded test B: {test_b.shape}")
        
        # Validate schema
        self._validate_schema(test_meta, test_a, test_b)
        
        return test_meta, test_a, test_b
    
    def _load_with_optimization(self, path: Path) -> pd.DataFrame:
        """
        Load CSV with memory optimization.
        
        Args:
            path: Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded and optimized data
        """
        if not path.exists():
            logger.warning(f"File not found: {path}, returning empty DataFrame")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        
        # Optimize data types for memory efficiency
        for col in df.columns:
            if df[col].dtype == 'int64':
                # Check if values fit in int32
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
            elif df[col].dtype == 'float64':
                # Convert to float32 for memory efficiency
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'object':
                # Check if categorical
                if df[col].nunique() / len(df) < 0.5:
                    df[col] = df[col].astype('category')
        
        return df
    
    def _validate_schema(self, test_meta: pd.DataFrame, test_a: pd.DataFrame, test_b: pd.DataFrame) -> None:
        """
        Validate data schema.
        
        Args:
            test_meta: Test metadata DataFrame
            test_a: Test A DataFrame
            test_b: Test B DataFrame
            
        Raises:
            ValueError: If schema validation fails
        """
        # Validate test_meta
        required_cols = ['Test_id', 'Test']
        missing_cols = [col for col in required_cols if col not in test_meta.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns in test_meta: {missing_cols}")
        
        # Check for null Test_id
        if test_meta['Test_id'].isnull().any():
            raise ValueError("Found null values in Test_id")
        
        # Check Test values
        valid_tests = {'A', 'B'}
        invalid_tests = set(test_meta['Test'].unique()) - valid_tests
        if invalid_tests:
            raise ValueError(f"Invalid Test values: {invalid_tests}")
        
        logger.info("Schema validation passed")
    
    def merge_test_data(self, test_meta: pd.DataFrame, test_a: pd.DataFrame, test_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Merge test metadata with detailed test data.
        
        Args:
            test_meta: Test metadata
            test_a: Test A details
            test_b: Test B details
            
        Returns:
            tuple: (merged_a, merged_b) DataFrames
        """
        # Extract A and B metadata
        meta_a = test_meta[test_meta['Test'] == 'A'][['Test_id', 'Test']].copy()
        meta_b = test_meta[test_meta['Test'] == 'B'][['Test_id', 'Test']].copy()
        
        # Merge with detailed data
        if len(test_a) > 0:
            merged_a = meta_a.merge(test_a, on='Test_id', how='left')
            logger.info(f"Merged test A: {merged_a.shape}")
        else:
            merged_a = pd.DataFrame()
        
        if len(test_b) > 0:
            merged_b = meta_b.merge(test_b, on='Test_id', how='left')
            logger.info(f"Merged test B: {merged_b.shape}")
        else:
            merged_b = pd.DataFrame()
        
        return merged_a, merged_b
