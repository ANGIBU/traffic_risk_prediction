# src/features.py
# Feature engineering for cognitive test data

import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline with domain knowledge.
    Simplified version to reduce overfitting.
    """
    
    def __init__(self, config):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.eps = config.preprocessing['eps']
        
    def _ensure_numeric(self, series):
        """
        Ensure series is numeric, converting from category if needed.
        
        Args:
            series: Pandas Series
            
        Returns:
            pd.Series: Numeric series
        """
        if pd.api.types.is_categorical_dtype(series):
            return pd.to_numeric(series.astype(str), errors='coerce')
        elif pd.api.types.is_object_dtype(series):
            return pd.to_numeric(series, errors='coerce')
        return series
    
    def _safe_multiply(self, a, b):
        """
        Safe multiplication handling category dtypes.
        
        Args:
            a: First operand
            b: Second operand
            
        Returns:
            Result of multiplication
        """
        a_num = self._ensure_numeric(a)
        b_num = self._ensure_numeric(b)
        return a_num * b_num
    
    def _safe_subtract(self, a, b):
        """
        Safe subtraction handling category dtypes.
        
        Args:
            a: First operand
            b: Second operand
            
        Returns:
            Result of subtraction
        """
        a_num = self._ensure_numeric(a)
        b_num = self._ensure_numeric(b)
        return a_num - b_num
    
    def _safe_add(self, a, b):
        """
        Safe addition handling category dtypes.
        
        Args:
            a: First operand
            b: Second operand
            
        Returns:
            Result of addition
        """
        a_num = self._ensure_numeric(a)
        b_num = self._ensure_numeric(b)
        return a_num + b_num
    
    def add_features_a(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for test type A.
        Simplified version to reduce overfitting and maintain 57 features target.
        
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
        
        # Convert all numeric columns from category to numeric
        for col in feats.columns:
            if pd.api.types.is_categorical_dtype(feats[col]):
                feats[col] = pd.to_numeric(feats[col].astype(str), errors='ignore')
        
        # Age binning for non-linear relationship
        if "Age_num" in feats.columns:
            age_num = self._ensure_numeric(feats["Age_num"])
            feats["Age_bin_20s"] = (age_num < 30).astype(float)
            feats["Age_bin_30s"] = ((age_num >= 30) & (age_num < 40)).astype(float)
            feats["Age_bin_40s"] = ((age_num >= 40) & (age_num < 50)).astype(float)
            feats["Age_bin_50plus"] = (age_num >= 50).astype(float)
        
        # Year-Month index
        if self._has(feats, ["Year", "Month"]):
            feats["YearMonthIndex"] = self._safe_multiply(feats["Year"], 12) + self._ensure_numeric(feats["Month"])
        
        # Basic speed-accuracy tradeoff features
        if self._has(feats, ["A1_rt_mean", "A1_resp_rate"]):
            feats["A1_speed_acc_tradeoff"] = self._safe_div(feats["A1_rt_mean"], feats["A1_resp_rate"], eps)
        if self._has(feats, ["A2_rt_mean", "A2_resp_rate"]):
            feats["A2_speed_acc_tradeoff"] = self._safe_div(feats["A2_rt_mean"], feats["A2_resp_rate"], eps)
        if self._has(feats, ["A4_rt_mean", "A4_acc_rate"]):
            feats["A4_speed_acc_tradeoff"] = self._safe_div(feats["A4_rt_mean"], feats["A4_acc_rate"], eps)
        
        # Coefficient of variation for reaction time
        for k in ["A1", "A3", "A4"]:
            m, s = f"{k}_rt_mean", f"{k}_rt_std"
            if self._has(feats, [m, s]):
                feats[f"{k}_rt_cv"] = self._safe_div(feats[s], feats[m], eps)
        
        # Absolute differences for key comparisons
        for name, base in [
            ("A1_rt_side_gap_abs", "A1_rt_side_diff"),
            ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
            ("A4_stroop_gap_abs", "A4_stroop_diff"),
        ]:
            if base in feats.columns:
                feats[name] = self._ensure_numeric(feats[base]).abs()
        
        # Temporal adaptation features
        if self._has(feats, ["A1_rt_first_half", "A1_rt_second_half"]):
            feats["A1_rt_adaptation"] = self._safe_subtract(feats["A1_rt_second_half"], feats["A1_rt_first_half"])
        
        if self._has(feats, ["A2_rt_first_half", "A2_rt_second_half"]):
            feats["A2_rt_adaptation"] = self._safe_subtract(feats["A2_rt_second_half"], feats["A2_rt_first_half"])
        
        if self._has(feats, ["A4_rt_first_half", "A4_rt_second_half"]):
            feats["A4_rt_adaptation"] = self._safe_subtract(feats["A4_rt_second_half"], feats["A4_rt_first_half"])
        
        # Extreme performance indicators
        if self._has(feats, ["A1_rt_range", "A1_rt_mean"]):
            feats["A1_rt_extreme_ratio"] = self._safe_div(feats["A1_rt_range"], feats["A1_rt_mean"], eps)
        
        if self._has(feats, ["A3_rt_range", "A3_rt_mean"]):
            feats["A3_rt_extreme_ratio"] = self._safe_div(feats["A3_rt_range"], feats["A3_rt_mean"], eps)
        
        # Cross-test performance comparison
        if self._has(feats, ["A1_rt_mean", "A4_rt_mean"]):
            feats["A1_A4_rt_ratio"] = self._safe_div(feats["A1_rt_mean"], feats["A4_rt_mean"], eps)
        
        if self._has(feats, ["A3_rt_mean", "A4_rt_mean"]):
            feats["A3_A4_rt_ratio"] = self._safe_div(feats["A3_rt_mean"], feats["A4_rt_mean"], eps)
        
        # Median-based robustness
        if self._has(feats, ["A3_rt_median", "A3_rt_mean"]):
            feats["A3_rt_robustness"] = self._safe_div(feats["A3_rt_median"], feats["A3_rt_mean"], eps)
        
        if self._has(feats, ["A4_rt_median", "A4_rt_mean"]):
            feats["A4_rt_robustness"] = self._safe_div(feats["A4_rt_median"], feats["A4_rt_mean"], eps)
        
        # Overall response quality
        acc_cols = [c for c in feats.columns if 'acc_rate' in c or 'resp_rate' in c]
        if len(acc_cols) > 0:
            acc_df = feats[acc_cols].apply(self._ensure_numeric)
            feats["A_overall_acc"] = acc_df.mean(axis=1)
            feats["A_overall_acc_min"] = acc_df.min(axis=1)
        
        # Overall reaction time profile
        rt_mean_cols = [c for c in feats.columns if c.endswith('_rt_mean') and c.startswith('A')]
        if len(rt_mean_cols) > 0:
            rt_df = feats[rt_mean_cols].apply(self._ensure_numeric)
            feats["A_overall_rt_mean"] = rt_df.mean(axis=1)
            feats["A_overall_rt_std"] = rt_df.std(axis=1)
        
        # Overall consistency profile
        consistency_cols = [c for c in feats.columns if 'consistency' in c and c.startswith('A')]
        if len(consistency_cols) > 0:
            cons_df = feats[consistency_cols].apply(self._ensure_numeric)
            feats["A_overall_consistency"] = cons_df.mean(axis=1)
        
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Feature engineering complete for test A: {feats.shape}")
        return feats
    
    def add_features_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for test type B.
        Enhanced with 20 additional Type B specific features.
        
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
        
        # Convert all numeric columns from category to numeric
        for col in feats.columns:
            if pd.api.types.is_categorical_dtype(feats[col]):
                feats[col] = pd.to_numeric(feats[col].astype(str), errors='ignore')
        
        # Age binning for non-linear relationship
        if "Age_num" in feats.columns:
            age_num = self._ensure_numeric(feats["Age_num"])
            feats["Age_bin_20s"] = (age_num < 30).astype(float)
            feats["Age_bin_30s"] = ((age_num >= 30) & (age_num < 40)).astype(float)
            feats["Age_bin_40s"] = ((age_num >= 40) & (age_num < 50)).astype(float)
            feats["Age_bin_50plus"] = (age_num >= 50).astype(float)
        
        # Year-Month index
        if self._has(feats, ["Year", "Month"]):
            feats["YearMonthIndex"] = self._safe_multiply(feats["Year"], 12) + self._ensure_numeric(feats["Month"])
        
        # Basic speed-accuracy tradeoff features
        for k, acc_col, rt_col in [
            ("B1", "B1_acc_task1", "B1_rt_mean"),
            ("B3", "B3_acc_rate", "B3_rt_mean"),
            ("B4", "B4_acc_rate", "B4_rt_mean"),
            ("B5", "B5_acc_rate", "B5_rt_mean"),
        ]:
            if self._has(feats, [rt_col, acc_col]):
                feats[f"{k}_speed_acc_tradeoff"] = self._safe_div(feats[rt_col], feats[acc_col], eps)
        
        # Coefficient of variation for reaction time
        for k in ["B1", "B3", "B5"]:
            m, s = f"{k}_rt_mean", f"{k}_rt_std"
            if self._has(feats, [m, s]):
                feats[f"{k}_rt_cv"] = self._safe_div(feats[s], feats[m], eps)
        
        # Temporal adaptation features
        if self._has(feats, ["B1_rt_first_half", "B1_rt_second_half"]):
            feats["B1_rt_adaptation"] = self._safe_subtract(feats["B1_rt_second_half"], feats["B1_rt_first_half"])
        
        if self._has(feats, ["B2_rt_first_half", "B2_rt_second_half"]):
            feats["B2_rt_adaptation"] = self._safe_subtract(feats["B2_rt_second_half"], feats["B2_rt_first_half"])
        
        # Extreme performance indicators
        if self._has(feats, ["B1_rt_range", "B1_rt_mean"]):
            feats["B1_rt_extreme_ratio"] = self._safe_div(feats["B1_rt_range"], feats["B1_rt_mean"], eps)
        
        if self._has(feats, ["B3_rt_range", "B3_rt_mean"]):
            feats["B3_rt_extreme_ratio"] = self._safe_div(feats["B3_rt_range"], feats["B3_rt_mean"], eps)
        
        # Cross-test performance comparison
        if self._has(feats, ["B1_rt_mean", "B2_rt_mean"]):
            feats["B1_B2_rt_ratio"] = self._safe_div(feats["B1_rt_mean"], feats["B2_rt_mean"], eps)
        
        if self._has(feats, ["B4_rt_mean", "B5_rt_mean"]):
            feats["B4_B5_rt_ratio"] = self._safe_div(feats["B4_rt_mean"], feats["B5_rt_mean"], eps)
        
        # Cross-test consistency comparison
        if self._has(feats, ["B1_rt_consistency", "B2_rt_consistency"]):
            feats["B1_B2_consistency_diff"] = self._safe_subtract(feats["B1_rt_consistency"], feats["B2_rt_consistency"])
        
        # Task consistency within tests
        if self._has(feats, ["B1_acc_task1", "B1_acc_task2"]):
            feats["B1_task_consistency"] = 1.0 - self._safe_subtract(feats["B1_acc_task1"], feats["B1_acc_task2"]).abs()
        
        if self._has(feats, ["B2_acc_task1", "B2_acc_task2"]):
            feats["B2_task_consistency"] = 1.0 - self._safe_subtract(feats["B2_acc_task1"], feats["B2_acc_task2"]).abs()
        
        # B1-B2 performance gap analysis
        if self._has(feats, ["B1_acc_task1", "B2_acc_task1"]):
            b1_acc = self._ensure_numeric(feats["B1_acc_task1"])
            b2_acc = self._ensure_numeric(feats["B2_acc_task1"])
            feats["B1_B2_acc_gap"] = b2_acc - b1_acc
        
        if self._has(feats, ["B1_rt_mean", "B2_rt_mean"]):
            b1_rt = self._ensure_numeric(feats["B1_rt_mean"])
            b2_rt = self._ensure_numeric(feats["B2_rt_mean"])
            feats["B1_B2_rt_gap"] = b2_rt - b1_rt
        
        # B3-B4-B5 sequential performance pattern
        if self._has(feats, ["B3_acc_rate", "B4_acc_rate", "B5_acc_rate"]):
            b3_acc = self._ensure_numeric(feats["B3_acc_rate"])
            b4_acc = self._ensure_numeric(feats["B4_acc_rate"])
            b5_acc = self._ensure_numeric(feats["B5_acc_rate"])
            feats["B3_B5_acc_gap"] = b3_acc - b5_acc
        
        if self._has(feats, ["B3_rt_mean", "B4_rt_mean", "B5_rt_mean"]):
            b3_rt = self._ensure_numeric(feats["B3_rt_mean"])
            b4_rt = self._ensure_numeric(feats["B4_rt_mean"])
            b5_rt = self._ensure_numeric(feats["B5_rt_mean"])
            feats["B3_B5_rt_gap"] = b5_rt - b3_rt
        
        # Median-based robustness
        if self._has(feats, ["B3_rt_median", "B3_rt_mean"]):
            feats["B3_rt_robustness"] = self._safe_div(feats["B3_rt_median"], feats["B3_rt_mean"], eps)
        
        # === NEW TYPE B SPECIFIC FEATURES (20 features) ===
        
        # 1. B1-B2 Learning Effect Analysis (4 features)
        # Measures adaptation and learning between similar sequential tests
        if self._has(feats, ["B1_rt_mean", "B2_rt_mean"]):
            b1_rt = self._ensure_numeric(feats["B1_rt_mean"])
            b2_rt = self._ensure_numeric(feats["B2_rt_mean"])
            feats["B1_B2_learning_rate"] = self._safe_div(b2_rt - b1_rt, b1_rt, eps)
        
        if self._has(feats, ["B1_acc_task1", "B2_acc_task1"]):
            b1_acc = self._ensure_numeric(feats["B1_acc_task1"])
            b2_acc = self._ensure_numeric(feats["B2_acc_task1"])
            feats["B1_B2_acc_learning"] = self._safe_div(b2_acc - b1_acc, b1_acc + eps, eps)
        
        if self._has(feats, ["B1_rt_consistency", "B2_rt_consistency"]):
            b1_cons = self._ensure_numeric(feats["B1_rt_consistency"])
            b2_cons = self._ensure_numeric(feats["B2_rt_consistency"])
            feats["B1_B2_consistency_learning"] = b2_cons - b1_cons
        
        if self._has(feats, ["B1_rt_adaptation", "B2_rt_adaptation"]):
            feats["B1_B2_adaptation_change"] = self._safe_subtract(feats["B2_rt_adaptation"], feats["B1_rt_adaptation"])
        
        # 2. B3-B4-B5 Sequential Pattern (4 features)
        # Analyzes progression through increasing complexity
        if self._has(feats, ["B3_rt_mean", "B4_rt_mean", "B5_rt_mean"]):
            b3_rt = self._ensure_numeric(feats["B3_rt_mean"])
            b4_rt = self._ensure_numeric(feats["B4_rt_mean"])
            b5_rt = self._ensure_numeric(feats["B5_rt_mean"])
            feats["B345_rt_progression"] = (b5_rt - b3_rt) / (b4_rt + eps)
            feats["B345_rt_variability"] = np.maximum(np.maximum(b3_rt, b4_rt), b5_rt) - np.minimum(np.minimum(b3_rt, b4_rt), b5_rt)
        
        if self._has(feats, ["B3_acc_rate", "B4_acc_rate", "B5_acc_rate"]):
            b3_acc = self._ensure_numeric(feats["B3_acc_rate"])
            b4_acc = self._ensure_numeric(feats["B4_acc_rate"])
            b5_acc = self._ensure_numeric(feats["B5_acc_rate"])
            feats["B345_acc_stability"] = 1.0 - (np.maximum(np.maximum(b3_acc, b4_acc), b5_acc) - np.minimum(np.minimum(b3_acc, b4_acc), b5_acc))
            feats["B345_acc_trend_slope"] = (b5_acc - b3_acc) / 2.0
        
        # 3. B6-B7-B8 Simple Response Consistency (3 features)
        # Measures stability in simple reaction tests
        if self._has(feats, ["B6_acc_rate", "B7_acc_rate", "B8_acc_rate"]):
            b6_acc = self._ensure_numeric(feats["B6_acc_rate"])
            b7_acc = self._ensure_numeric(feats["B7_acc_rate"])
            b8_acc = self._ensure_numeric(feats["B8_acc_rate"])
            feats["B678_simple_consistency"] = 1.0 - (np.maximum(np.maximum(b6_acc, b7_acc), b8_acc) - np.minimum(np.minimum(b6_acc, b7_acc), b8_acc))
            feats["B678_simple_mean"] = (b6_acc + b7_acc + b8_acc) / 3.0
            feats["B678_simple_std"] = np.sqrt(((b6_acc - feats["B678_simple_mean"])**2 + 
                                                  (b7_acc - feats["B678_simple_mean"])**2 + 
                                                  (b8_acc - feats["B678_simple_mean"])**2) / 3.0)
        
        # 4. Cross-Test Interaction Features (5 features)
        # Captures relationships between different test types
        if self._has(feats, ["B1_rt_mean", "B3_rt_mean", "B5_rt_mean"]):
            b1_rt = self._ensure_numeric(feats["B1_rt_mean"])
            b3_rt = self._ensure_numeric(feats["B3_rt_mean"])
            b5_rt = self._ensure_numeric(feats["B5_rt_mean"])
            feats["B135_rt_balance"] = self._safe_div(b3_rt, (b1_rt + b5_rt) / 2.0 + eps, eps)
        
        if self._has(feats, ["B2_rt_mean", "B4_rt_mean"]):
            b2_rt = self._ensure_numeric(feats["B2_rt_mean"])
            b4_rt = self._ensure_numeric(feats["B4_rt_mean"])
            feats["B24_rt_ratio"] = self._safe_div(b2_rt, b4_rt, eps)
        
        if self._has(feats, ["B1_acc_task1", "B5_acc_rate"]):
            feats["B1_B5_acc_consistency"] = 1.0 - abs(self._safe_subtract(feats["B1_acc_task1"], feats["B5_acc_rate"]))
        
        if self._has(feats, ["B3_rt_consistency", "B6_acc_consistency"]):
            feats["B3_B6_consistency_product"] = self._safe_multiply(feats["B3_rt_consistency"], feats["B6_acc_consistency"])
        
        if self._has(feats, ["B4_acc_rate", "B8_acc_rate"]):
            feats["B4_B8_simple_complex_gap"] = self._safe_subtract(feats["B8_acc_rate"], feats["B4_acc_rate"])
        
        # 5. Non-linear Transformations (4 features)
        # Captures non-linear relationships in key metrics
        if self._has(feats, ["B1_rt_mean"]):
            b1_rt = self._ensure_numeric(feats["B1_rt_mean"])
            feats["B1_rt_log"] = np.log1p(b1_rt)
            feats["B1_rt_sqrt"] = np.sqrt(b1_rt + eps)
        
        if self._has(feats, ["B3_rt_std"]):
            b3_std = self._ensure_numeric(feats["B3_rt_std"])
            feats["B3_rt_std_sqrt"] = np.sqrt(b3_std + eps)
        
        if self._has(feats, ["B5_acc_rate"]):
            b5_acc = self._ensure_numeric(feats["B5_acc_rate"])
            feats["B5_acc_squared"] = b5_acc ** 2
        
        # === END OF NEW TYPE B SPECIFIC FEATURES ===
        
        # Overall response quality
        acc_cols = [c for c in feats.columns if 'acc_' in c and c.startswith('B')]
        if len(acc_cols) > 0:
            acc_df = feats[acc_cols].apply(self._ensure_numeric)
            feats["B_overall_acc"] = acc_df.mean(axis=1)
            feats["B_overall_acc_std"] = acc_df.std(axis=1)
            feats["B_overall_acc_min"] = acc_df.min(axis=1)
        
        # Overall reaction time profile
        rt_mean_cols = [c for c in feats.columns if c.endswith('_rt_mean') and c.startswith('B')]
        if len(rt_mean_cols) > 0:
            rt_df = feats[rt_mean_cols].apply(self._ensure_numeric)
            feats["B_overall_rt_mean"] = rt_df.mean(axis=1)
            feats["B_overall_rt_std"] = rt_df.std(axis=1)
            feats["B_overall_rt_min"] = rt_df.min(axis=1)
        
        # Overall consistency profile
        consistency_cols = [c for c in feats.columns if 'consistency' in c and c.startswith('B')]
        if len(consistency_cols) > 0:
            cons_df = feats[consistency_cols].apply(self._ensure_numeric)
            feats["B_overall_consistency"] = cons_df.mean(axis=1)
            feats["B_overall_consistency_std"] = cons_df.std(axis=1)
        
        # Trend uniformity
        trend_cols = [c for c in feats.columns if 'trend' in c and c.startswith('B')]
        if len(trend_cols) > 0:
            trend_df = feats[trend_cols].apply(self._ensure_numeric)
            feats["B_overall_trend_mean"] = trend_df.mean(axis=1)
        
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
        a_num = self._ensure_numeric(a)
        b_num = self._ensure_numeric(b)
        return a_num / (b_num + eps)