# src/predictor.py
# Model inference engine with batch processing

import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict
import lightgbm as lgb

logger = logging.getLogger(__name__)

class Predictor:
    """
    Model inference engine.
    
    Handles model loading, batch prediction, and feature alignment
    for optimized inference performance.
    """
    
    def __init__(self, config):
        """
        Initialize predictor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.inference_config = config.inference
        self.model_a = None
        self.model_b = None
        self.feature_names_a = None
        self.feature_names_b = None
        
        # Drop columns that should not be used as features
        self.drop_cols = ["Test_id", "Test", "PrimaryKey", "Age", "TestDate"]
        
        # Load models
        self._load_models()
    
    def _load_models(self) -> None:
        """Load trained models from disk."""
        model_a_path = self.config.paths['model']
        model_b_path = self.config.paths['model_b']
        
        # Load model A
        if model_a_path.exists():
            logger.info(f"Loading model A from {model_a_path}")
            self.model_a = joblib.load(model_a_path)
            
            if hasattr(self.model_a, 'feature_name_'):
                self.feature_names_a = list(self.model_a.feature_name_)
                logger.info(f"Model A expects {len(self.feature_names_a)} features")
            
            logger.info("Model A loaded successfully")
        else:
            logger.warning(f"Model A not found at {model_a_path}")
        
        # Load model B
        if model_b_path.exists():
            logger.info(f"Loading model B from {model_b_path}")
            self.model_b = joblib.load(model_b_path)
            
            if hasattr(self.model_b, 'feature_name_'):
                self.feature_names_b = list(self.model_b.feature_name_)
                logger.info(f"Model B expects {len(self.feature_names_b)} features")
            
            logger.info("Model B loaded successfully")
        else:
            logger.warning(f"Model B not found at {model_b_path}")
    
    def predict(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate predictions for both test types.
        
        Args:
            df_a: Preprocessed and feature-engineered type A data
            df_b: Preprocessed and feature-engineered type B data
            
        Returns:
            dict: Dictionary with 'A' and 'B' keys containing prediction arrays
        """
        predictions = {}
        
        # Predict for type A
        if len(df_a) > 0 and self.model_a is not None:
            logger.info(f"Predicting for {len(df_a)} type A samples")
            X_a = self._align_to_model(df_a, self.model_a, self.feature_names_a)
            pred_a = self._batch_predict(self.model_a, X_a)
            predictions['A'] = pred_a
            logger.info(f"Type A predictions: mean={pred_a.mean():.4f}, std={pred_a.std():.4f}")
        else:
            predictions['A'] = np.array([])
        
        # Predict for type B
        if len(df_b) > 0 and self.model_b is not None:
            logger.info(f"Predicting for {len(df_b)} type B samples")
            X_b = self._align_to_model(df_b, self.model_b, self.feature_names_b)
            pred_b = self._batch_predict(self.model_b, X_b)
            predictions['B'] = pred_b
            logger.info(f"Type B predictions: mean={pred_b.mean():.4f}, std={pred_b.std():.4f}")
        else:
            predictions['B'] = np.array([])
        
        return predictions
    
    def _align_to_model(self, df: pd.DataFrame, model, feature_names: Optional[list]) -> pd.DataFrame:
        """
        Align features to model expectations.
        
        Args:
            df: DataFrame with features
            model: Trained model
            feature_names: Expected feature names
            
        Returns:
            pd.DataFrame: Aligned feature matrix
        """
        # Drop non-feature columns
        X = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore").copy()
        
        if feature_names is None or len(feature_names) == 0:
            # Fallback: use all numeric columns
            X = X.select_dtypes(include=[np.number]).copy()
            return X.fillna(0.0)
        
        # Add missing features with zeros
        for c in feature_names:
            if c not in X.columns:
                X[c] = 0.0
        
        # Select and order features according to model
        X = X[feature_names]
        
        # Convert to numeric and fill NaN
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        
        return X
    
    def _batch_predict(self, model, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with batch processing.
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            np.ndarray: Predicted probabilities (0-1)
        """
        batch_size = self.inference_config['batch_size']
        n_samples = len(X)
        predictions = np.zeros(n_samples, dtype=np.float32)
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        logger.info(f"Processing {n_samples} samples in {n_batches} batches")
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = X.iloc[i:end_idx]
            
            # Predict probabilities
            batch_pred = model.predict_proba(batch)[:, 1]
            predictions[i:end_idx] = batch_pred
            
            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"Processed batch {i // batch_size + 1}/{n_batches}")
        
        # Clip to valid range
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
