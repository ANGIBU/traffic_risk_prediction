# src/predictor.py
# Model inference engine with batch processing

import joblib
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, List, Union
import lightgbm as lgb

logger = logging.getLogger(__name__)

class Predictor:
    """
    Model inference engine.
    
    Handles model loading, batch prediction, and feature alignment
    for optimized inference performance. Supports both single models
    and ensemble models.
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
        self.calibrator_a = None
        self.calibrator_b = None
        self.feature_names_a = None
        self.feature_names_b = None
        self.use_calibration = config.training.get('use_calibration', False)
        
        # Drop columns that should not be used as features
        self.drop_cols = ["Test_id", "Test", "PrimaryKey", "Age", "TestDate"]
        
        # Load models and feature names
        self._load_models()
        self._load_calibrators()
        self._load_feature_names()
    
    def _load_models(self) -> None:
        """Load trained models from disk. Supports both single and ensemble models."""
        model_a_path = self.config.paths['model']
        model_b_path = self.config.paths['model_b']
        
        # Load model A
        if model_a_path.exists():
            logger.info(f"Loading model A from {model_a_path}")
            loaded_model = joblib.load(model_a_path)
            # Check if it's a list (ensemble) or single model
            if isinstance(loaded_model, list):
                self.model_a = loaded_model
                logger.info(f"Model A loaded as ensemble with {len(self.model_a)} models")
            else:
                self.model_a = loaded_model
                logger.info("Model A loaded successfully")
        else:
            logger.warning(f"Model A not found at {model_a_path}")
        
        # Load model B
        if model_b_path.exists():
            logger.info(f"Loading model B from {model_b_path}")
            loaded_model = joblib.load(model_b_path)
            # Check if it's a list (ensemble) or single model
            if isinstance(loaded_model, list):
                self.model_b = loaded_model
                logger.info(f"Model B loaded as ensemble with {len(self.model_b)} models")
            else:
                self.model_b = loaded_model
                logger.info("Model B loaded successfully")
        else:
            logger.warning(f"Model B not found at {model_b_path}")
    
    def _load_calibrators(self) -> None:
        """Load calibrators from disk if available."""
        if not self.use_calibration:
            logger.info("Calibration disabled in config")
            return
        
        calibrator_a_path = self.config.paths['model_dir'] / 'calibrator_A.pkl'
        calibrator_b_path = self.config.paths['model_dir'] / 'calibrator_B.pkl'
        
        if calibrator_a_path.exists():
            logger.info(f"Loading calibrator A from {calibrator_a_path}")
            self.calibrator_a = joblib.load(calibrator_a_path)
            logger.info("Calibrator A loaded successfully")
        else:
            logger.info(f"Calibrator A not found at {calibrator_a_path}")
        
        if calibrator_b_path.exists():
            logger.info(f"Loading calibrator B from {calibrator_b_path}")
            self.calibrator_b = joblib.load(calibrator_b_path)
            logger.info("Calibrator B loaded successfully")
        else:
            logger.info(f"Calibrator B not found at {calibrator_b_path}")
    
    def _load_feature_names(self) -> None:
        """Load feature names from disk."""
        feature_names_a_path = self.config.paths.get('feature_names_a')
        feature_names_b_path = self.config.paths.get('feature_names_b')
        
        # Load feature names for model A
        if feature_names_a_path and feature_names_a_path.exists():
            logger.info(f"Loading feature names A from {feature_names_a_path}")
            with open(feature_names_a_path, 'r') as f:
                self.feature_names_a = [line.strip() for line in f if line.strip()]
            logger.info(f"Model A expects {len(self.feature_names_a)} features")
        elif self.model_a:
            # Try to get from model
            if isinstance(self.model_a, list):
                self.feature_names_a = list(self.model_a[0].feature_name_())
            elif hasattr(self.model_a, 'feature_name_'):
                self.feature_names_a = list(self.model_a.feature_name_())
            if self.feature_names_a:
                logger.info(f"Model A feature names from model: {len(self.feature_names_a)} features")
        else:
            logger.warning("Feature names A not found")
        
        # Load feature names for model B
        if feature_names_b_path and feature_names_b_path.exists():
            logger.info(f"Loading feature names B from {feature_names_b_path}")
            with open(feature_names_b_path, 'r') as f:
                self.feature_names_b = [line.strip() for line in f if line.strip()]
            logger.info(f"Model B expects {len(self.feature_names_b)} features")
        elif self.model_b:
            # Try to get from model
            if isinstance(self.model_b, list):
                self.feature_names_b = list(self.model_b[0].feature_name_())
            elif hasattr(self.model_b, 'feature_name_'):
                self.feature_names_b = list(self.model_b.feature_name_())
            if self.feature_names_b:
                logger.info(f"Model B feature names from model: {len(self.feature_names_b)} features")
        else:
            logger.warning("Feature names B not found")
    
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
            X_a = self._align_to_model(df_a, self.model_a, self.feature_names_a, 'A')
            pred_a = self._batch_predict(self.model_a, X_a)
            
            # Apply calibration if available
            if self.use_calibration and self.calibrator_a is not None:
                logger.info("Applying calibration to Type A predictions")
                pred_a = self.calibrator_a.transform(pred_a)
                pred_a = np.clip(pred_a, 0.0, 1.0)
            
            predictions['A'] = pred_a
            logger.info(f"Type A predictions: mean={pred_a.mean():.4f}, std={pred_a.std():.4f}")
        else:
            predictions['A'] = np.array([])
        
        # Predict for type B
        if len(df_b) > 0 and self.model_b is not None:
            logger.info(f"Predicting for {len(df_b)} type B samples")
            X_b = self._align_to_model(df_b, self.model_b, self.feature_names_b, 'B')
            pred_b = self._batch_predict(self.model_b, X_b)
            
            # Apply calibration if available
            if self.use_calibration and self.calibrator_b is not None:
                logger.info("Applying calibration to Type B predictions")
                pred_b = self.calibrator_b.transform(pred_b)
                pred_b = np.clip(pred_b, 0.0, 1.0)
            
            predictions['B'] = pred_b
            logger.info(f"Type B predictions: mean={pred_b.mean():.4f}, std={pred_b.std():.4f}")
        else:
            predictions['B'] = np.array([])
        
        return predictions
    
    def _align_to_model(self, df: pd.DataFrame, model: Union[object, List], 
                        feature_names: Optional[List[str]], test_type: str) -> pd.DataFrame:
        """
        Align features to model expectations.
        
        Args:
            df: DataFrame with features
            model: Trained model (single or list for ensemble)
            feature_names: Expected feature names
            test_type: 'A' or 'B' for logging
            
        Returns:
            pd.DataFrame: Aligned feature matrix
        """
        # Drop non-feature columns
        X = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore").copy()
        
        if feature_names is None or len(feature_names) == 0:
            logger.warning(f"No feature names provided for type {test_type}, using all numeric columns")
            X = X.select_dtypes(include=[np.number]).copy()
            return X.fillna(0.0)
        
        logger.info(f"Type {test_type}: Aligning {len(X.columns)} generated features to {len(feature_names)} model features")
        
        # Check which features are missing and which are extra
        generated_features = set(X.columns)
        expected_features = set(feature_names)
        
        missing_features = expected_features - generated_features
        extra_features = generated_features - expected_features
        
        if missing_features:
            logger.warning(f"Type {test_type}: {len(missing_features)} features missing (will be filled with 0)")
            logger.debug(f"Missing features: {sorted(list(missing_features))[:10]}...")
        
        if extra_features:
            logger.info(f"Type {test_type}: {len(extra_features)} extra features (will be ignored)")
        
        # Add missing features with zeros
        for feat in missing_features:
            X[feat] = 0.0
        
        # Select and order features according to model
        X = X[feature_names].copy()
        
        # Convert to numeric and fill NaN
        for col in X.columns:
            if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(0.0)
        
        # Final validation
        assert X.shape[1] == len(feature_names), f"Feature count mismatch: {X.shape[1]} vs {len(feature_names)}"
        assert list(X.columns) == feature_names, "Feature order mismatch"
        
        logger.info(f"Type {test_type}: Feature alignment complete - {X.shape}")
        
        return X
    
    def _batch_predict(self, model: Union[object, List], X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions with batch processing. Supports ensemble.
        
        Args:
            model: Trained model (single or list for ensemble)
            X: Feature matrix
            
        Returns:
            np.ndarray: Predicted probabilities (0-1)
        """
        batch_size = self.inference_config['batch_size']
        n_samples = len(X)
        predictions = np.zeros(n_samples, dtype=np.float32)
        
        n_batches = (n_samples + batch_size - 1) // batch_size
        logger.info(f"Processing {n_samples} samples in {n_batches} batches")
        
        # Check if ensemble
        is_ensemble = isinstance(model, list)
        if is_ensemble:
            logger.info(f"Using ensemble with {len(model)} models")
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch = X.iloc[i:end_idx]
            
            if is_ensemble:
                # Average predictions from all models
                batch_pred = np.zeros(len(batch), dtype=np.float32)
                for m in model:
                    if hasattr(m, 'predict_proba'):
                        batch_pred += m.predict_proba(batch)[:, 1]
                    else:
                        batch_pred += m.predict(batch)
                batch_pred /= len(model)
            else:
                # Single model prediction
                if hasattr(model, 'predict_proba'):
                    batch_pred = model.predict_proba(batch)[:, 1]
                else:
                    batch_pred = model.predict(batch)
            
            predictions[i:end_idx] = batch_pred
            
            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"Processed batch {i // batch_size + 1}/{n_batches}")
        
        # Clip to valid range
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions