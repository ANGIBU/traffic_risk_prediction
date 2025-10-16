# src/features.py
# Feature engineering for cognitive test data

import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline.
    
    Generates derived features from raw cognitive test data including
    statistical aggregations, rolling window features, and domain-specific features.
    """
    
    def __init__(self, config):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.eps = config.preprocessing['eps']
        
    def add_features_a(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for test type A.
        
        Args:
            df: Preprocessed type A data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        if len(df) == 0:
            return df
        
        feats = df.copy()
        eps = self.eps
        
        logger.info("Generating derived features for test A")
        
        # Year-Month index
        if self._has(feats, ["Year", "Month"]):
            feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]
        
        # Speed-accuracy tradeoff features
        if self._has(feats, ["A1_rt_mean", "A1_resp_rate"]):
            feats["A1_speed_acc_tradeoff"] = self._safe_div(feats["A1_rt_mean"], feats["A1_resp_rate"], eps)
        if self._has(feats, ["A2_rt_mean", "A2_resp_rate"]):
            feats["A2_speed_acc_tradeoff"] = self._safe_div(feats["A2_rt_mean"], feats["A2_resp_rate"], eps)
        if self._has(feats, ["A4_rt_mean", "A4_acc_rate"]):
            feats["A4_speed_acc_tradeoff"] = self._safe_div(feats["A4_rt_mean"], feats["A4_acc_rate"], eps)
        
        # Coefficient of variation (CV) for reaction time
        for k in ["A1", "A2", "A3", "A4"]:
            m, s = f"{k}_rt_mean", f"{k}_rt_std"
            if self._has(feats, [m, s]):
                feats[f"{k}_rt_cv"] = self._safe_div(feats[s], feats[m], eps)
        
        # Absolute differences
        for name, base in [
            ("A1_rt_side_gap_abs", "A1_rt_side_diff"),
            ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"),
            ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
            ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"),
            ("A4_stroop_gap_abs", "A4_stroop_diff"),
            ("A4_color_gap_abs", "A4_rt_color_diff"),
        ]:
            if base in feats.columns:
                feats[name] = feats[base].abs()
        
        # Ratio features
        if self._has(feats, ["A3_valid_ratio", "A3_invalid_ratio"]):
            feats["A3_valid_invalid_gap"] = feats["A3_valid_ratio"] - feats["A3_invalid_ratio"]
        if self._has(feats, ["A3_correct_ratio", "A3_invalid_ratio"]):
            feats["A3_correct_invalid_gap"] = feats["A3_correct_ratio"] - feats["A3_invalid_ratio"]
        if self._has(feats, ["A5_acc_change", "A5_acc_nonchange"]):
            feats["A5_change_nonchange_gap"] = feats["A5_acc_change"] - feats["A5_acc_nonchange"]
        
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Feature engineering complete for test A: {feats.shape}")
        return feats
    
    def add_features_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for test type B.
        
        Args:
            df: Preprocessed type B data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        if len(df) == 0:
            return df
        
        feats = df.copy()
        eps = self.eps
        
        logger.info("Generating derived features for test B")
        
        # Year-Month index
        if self._has(feats, ["Year", "Month"]):
            feats["YearMonthIndex"] = feats["Year"] * 12 + feats["Month"]
        
        # Speed-accuracy tradeoff features
        for k, acc_col, rt_col in [
            ("B1", "B1_acc_task1", "B1_rt_mean"),
            ("B2", "B2_acc_task1", "B2_rt_mean"),
            ("B3", "B3_acc_rate", "B3_rt_mean"),
            ("B4", "B4_acc_rate", "B4_rt_mean"),
            ("B5", "B5_acc_rate", "B5_rt_mean"),
        ]:
            if self._has(feats, [rt_col, acc_col]):
                feats[f"{k}_speed_acc_tradeoff"] = self._safe_div(feats[rt_col], feats[acc_col], eps)
        
        # Coefficient of variation (CV) for reaction time
        for k in ["B1", "B2", "B3", "B4", "B5"]:
            m, s = f"{k}_rt_mean", f"{k}_rt_std"
            if self._has(feats, [m, s]):
                feats[f"{k}_rt_cv"] = self._safe_div(feats[s], feats[m], eps)
        
        # Risk score composite
        parts = []
        for k in ["B4", "B5"]:
            if self._has(feats, [f"{k}_rt_cv"]):
                parts.append(0.25 * feats[f"{k}_rt_cv"].fillna(0))
        for k in ["B3", "B4", "B5"]:
            acc = f"{k}_acc_rate" if k not in ["B1", "B2"] else None
            if k in ["B1", "B2"]:
                acc = f"{k}_acc_task1"
            if acc in feats:
                parts.append(0.25 * (1 - feats[acc].fillna(0)))
        for k in ["B1", "B2"]:
            tcol = f"{k}_speed_acc_tradeoff"
            if tcol in feats:
                parts.append(0.25 * feats[tcol].fillna(0))
        if parts:
            feats["RiskScore_B"] = sum(parts)
        
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Feature engineering complete for test B: {feats.shape}")
        return feats
    
    def _has(self, df: pd.DataFrame, cols: List[str]) -> bool:
        """
        Check if DataFrame has all specified columns.
        
        Args:
            df: DataFrame to check
            cols: List of column names
            
        Returns:
            bool: True if all columns exist
        """
        return all(c in df.columns for c in cols)
    
    def _safe_div(self, a, b, eps: float = 1e-6):
        """
        Safe division with epsilon to avoid division by zero.
        
        Args:
            a: Numerator
            b: Denominator
            eps: Small epsilon value
            
        Returns:
            Result of division
        """
        return a / (b + eps)
