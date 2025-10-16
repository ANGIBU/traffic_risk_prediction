# src/features.py
# Feature engineering for cognitive test data

import pandas as pd
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline with domain knowledge and interaction features.
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
        Add derived features for test type A with domain knowledge.
        
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
                feats[name] = self._ensure_numeric(feats[base]).abs()
        
        # Ratio features
        if self._has(feats, ["A3_valid_ratio", "A3_invalid_ratio"]):
            feats["A3_valid_invalid_gap"] = self._safe_subtract(feats["A3_valid_ratio"], feats["A3_invalid_ratio"])
        if self._has(feats, ["A3_correct_ratio", "A3_invalid_ratio"]):
            feats["A3_correct_invalid_gap"] = self._safe_subtract(feats["A3_correct_ratio"], feats["A3_invalid_ratio"])
        if self._has(feats, ["A5_acc_change", "A5_acc_nonchange"]):
            feats["A5_change_nonchange_gap"] = self._safe_subtract(feats["A5_acc_change"], feats["A5_acc_nonchange"])
        
        # Temporal adaptation features
        if self._has(feats, ["A1_rt_first_half", "A1_rt_second_half"]):
            feats["A1_rt_adaptation"] = self._safe_subtract(feats["A1_rt_second_half"], feats["A1_rt_first_half"])
            feats["A1_rt_learning_rate"] = self._safe_div(
                self._safe_subtract(feats["A1_rt_first_half"], feats["A1_rt_second_half"]),
                feats["A1_rt_first_half"], eps
            )
        
        if self._has(feats, ["A2_rt_first_half", "A2_rt_second_half"]):
            feats["A2_rt_adaptation"] = self._safe_subtract(feats["A2_rt_second_half"], feats["A2_rt_first_half"])
        
        if self._has(feats, ["A4_rt_first_half", "A4_rt_second_half"]):
            feats["A4_rt_adaptation"] = self._safe_subtract(feats["A4_rt_second_half"], feats["A4_rt_first_half"])
        
        # Stability vs variability
        if self._has(feats, ["A1_rt_consistency", "A1_rt_std"]):
            feats["A1_stability_score"] = self._safe_multiply(
                feats["A1_rt_consistency"],
                (1.0 - self._ensure_numeric(feats["A1_rt_cv"]).fillna(0))
            )
        
        if self._has(feats, ["A2_rt_consistency", "A2_rt_std"]):
            feats["A2_stability_score"] = self._safe_multiply(
                feats["A2_rt_consistency"],
                (1.0 - self._ensure_numeric(feats["A2_rt_cv"]).fillna(0))
            )
        
        # Extreme performance indicators
        if self._has(feats, ["A1_rt_min", "A1_rt_max", "A1_rt_mean"]):
            feats["A1_rt_extreme_ratio"] = self._safe_div(
                self._safe_subtract(feats["A1_rt_max"], feats["A1_rt_min"]),
                feats["A1_rt_mean"], eps
            )
        
        if self._has(feats, ["A2_rt_range", "A2_rt_mean"]):
            feats["A2_rt_extreme_ratio"] = self._safe_div(feats["A2_rt_range"], feats["A2_rt_mean"], eps)
        
        if self._has(feats, ["A3_rt_range", "A3_rt_mean"]):
            feats["A3_rt_extreme_ratio"] = self._safe_div(feats["A3_rt_range"], feats["A3_rt_mean"], eps)
        
        if self._has(feats, ["A4_rt_range", "A4_rt_mean"]):
            feats["A4_rt_extreme_ratio"] = self._safe_div(feats["A4_rt_range"], feats["A4_rt_mean"], eps)
        
        # Cross-test performance comparison
        if self._has(feats, ["A1_rt_mean", "A2_rt_mean"]):
            feats["A1_A2_rt_ratio"] = self._safe_div(feats["A1_rt_mean"], feats["A2_rt_mean"], eps)
            feats["A1_A2_rt_diff"] = self._safe_subtract(feats["A1_rt_mean"], feats["A2_rt_mean"])
        
        if self._has(feats, ["A1_rt_mean", "A3_rt_mean"]):
            feats["A1_A3_rt_ratio"] = self._safe_div(feats["A1_rt_mean"], feats["A3_rt_mean"], eps)
        
        if self._has(feats, ["A1_rt_mean", "A4_rt_mean"]):
            feats["A1_A4_rt_ratio"] = self._safe_div(feats["A1_rt_mean"], feats["A4_rt_mean"], eps)
        
        if self._has(feats, ["A2_rt_mean", "A3_rt_mean"]):
            feats["A2_A3_rt_ratio"] = self._safe_div(feats["A2_rt_mean"], feats["A3_rt_mean"], eps)
        
        if self._has(feats, ["A2_rt_mean", "A4_rt_mean"]):
            feats["A2_A4_rt_ratio"] = self._safe_div(feats["A2_rt_mean"], feats["A4_rt_mean"], eps)
        
        if self._has(feats, ["A3_rt_mean", "A4_rt_mean"]):
            feats["A3_A4_rt_ratio"] = self._safe_div(feats["A3_rt_mean"], feats["A4_rt_mean"], eps)
        
        # Cross-test consistency comparison
        if self._has(feats, ["A1_rt_consistency", "A2_rt_consistency"]):
            feats["A1_A2_consistency_diff"] = self._safe_subtract(feats["A1_rt_consistency"], feats["A2_rt_consistency"])
        
        if self._has(feats, ["A1_rt_consistency", "A4_rt_consistency"]):
            feats["A1_A4_consistency_diff"] = self._safe_subtract(feats["A1_rt_consistency"], feats["A4_rt_consistency"])
        
        # Attention distribution features
        if self._has(feats, ["A1_rt_left", "A1_rt_right"]):
            total = self._safe_add(feats["A1_rt_left"], feats["A1_rt_right"])
            feats["A1_attention_asymmetry"] = self._safe_div(
                self._safe_subtract(feats["A1_rt_left"], feats["A1_rt_right"]).abs(),
                total, eps
            )
        
        # Stroop effect magnitude
        if self._has(feats, ["A4_stroop_diff", "A4_rt_mean"]):
            feats["A4_stroop_effect_ratio"] = self._safe_div(feats["A4_stroop_diff"], feats["A4_rt_mean"], eps)
        
        # Conditional variability ratio
        if self._has(feats, ["A1_rt_left_std", "A1_rt_right_std"]):
            feats["A1_side_variability_ratio"] = self._safe_div(
                feats["A1_rt_left_std"],
                feats["A1_rt_right_std"], eps
            )
        
        if self._has(feats, ["A1_rt_slow_std", "A1_rt_fast_std"]):
            feats["A1_speed_variability_ratio"] = self._safe_div(
                feats["A1_rt_slow_std"],
                feats["A1_rt_fast_std"], eps
            )
        
        # Median-based robustness
        if self._has(feats, ["A1_rt_median", "A1_rt_mean"]):
            feats["A1_rt_robustness"] = self._safe_div(feats["A1_rt_median"], feats["A1_rt_mean"], eps)
        
        if self._has(feats, ["A2_rt_median", "A2_rt_mean"]):
            feats["A2_rt_robustness"] = self._safe_div(feats["A2_rt_median"], feats["A2_rt_mean"], eps)
        
        if self._has(feats, ["A3_rt_median", "A3_rt_mean"]):
            feats["A3_rt_robustness"] = self._safe_div(feats["A3_rt_median"], feats["A3_rt_mean"], eps)
        
        if self._has(feats, ["A4_rt_median", "A4_rt_mean"]):
            feats["A4_rt_robustness"] = self._safe_div(feats["A4_rt_median"], feats["A4_rt_mean"], eps)
        
        # IQR-based stability
        if self._has(feats, ["A1_rt_iqr", "A1_rt_median"]):
            feats["A1_rt_iqr_ratio"] = self._safe_div(feats["A1_rt_iqr"], feats["A1_rt_median"], eps)
        
        if self._has(feats, ["A2_rt_iqr", "A2_rt_median"]):
            feats["A2_rt_iqr_ratio"] = self._safe_div(feats["A2_rt_iqr"], feats["A2_rt_median"], eps)
        
        # Outlier propensity
        if self._has(feats, ["A1_rt_outlier_count", "A1_rt_mean"]):
            feats["A1_outlier_rate"] = self._ensure_numeric(feats["A1_rt_outlier_count"]) / 20.0
        
        if self._has(feats, ["A3_rt_outlier_count", "A3_rt_mean"]):
            feats["A3_outlier_rate"] = self._ensure_numeric(feats["A3_rt_outlier_count"]) / 20.0
        
        # Skewness-based asymmetry
        if self._has(feats, ["A1_rt_skew"]):
            feats["A1_rt_skew_abs"] = self._ensure_numeric(feats["A1_rt_skew"]).abs()
        if self._has(feats, ["A2_rt_skew"]):
            feats["A2_rt_skew_abs"] = self._ensure_numeric(feats["A2_rt_skew"]).abs()
        if self._has(feats, ["A3_rt_skew"]):
            feats["A3_rt_skew_abs"] = self._ensure_numeric(feats["A3_rt_skew"]).abs()
        if self._has(feats, ["A4_rt_skew"]):
            feats["A4_rt_skew_abs"] = self._ensure_numeric(feats["A4_rt_skew"]).abs()
        
        # Overall response quality
        acc_cols = [c for c in feats.columns if 'acc_rate' in c or 'resp_rate' in c]
        if len(acc_cols) > 0:
            acc_df = feats[acc_cols].apply(self._ensure_numeric)
            feats["A_overall_acc"] = acc_df.mean(axis=1)
            feats["A_overall_acc_std"] = acc_df.std(axis=1)
        
        # Overall reaction time profile
        rt_mean_cols = [c for c in feats.columns if c.endswith('_rt_mean') and c.startswith('A')]
        if len(rt_mean_cols) > 0:
            rt_df = feats[rt_mean_cols].apply(self._ensure_numeric)
            feats["A_overall_rt_mean"] = rt_df.mean(axis=1)
            feats["A_overall_rt_std"] = rt_df.std(axis=1)
            feats["A_overall_rt_cv"] = self._safe_div(feats["A_overall_rt_std"], feats["A_overall_rt_mean"], eps)
        
        # Overall consistency profile
        consistency_cols = [c for c in feats.columns if 'consistency' in c and c.startswith('A')]
        if len(consistency_cols) > 0:
            cons_df = feats[consistency_cols].apply(self._ensure_numeric)
            feats["A_overall_consistency"] = cons_df.mean(axis=1)
            feats["A_overall_consistency_std"] = cons_df.std(axis=1)
        
        # Trend uniformity
        trend_cols = [c for c in feats.columns if 'trend' in c and c.startswith('A')]
        if len(trend_cols) > 0:
            trend_df = feats[trend_cols].apply(self._ensure_numeric)
            feats["A_overall_trend_mean"] = trend_df.mean(axis=1)
            feats["A_overall_trend_std"] = trend_df.std(axis=1)
        
        # Age interaction features
        if self._has(feats, ["Age_num", "A1_rt_mean"]):
            feats["Age_A1_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["A1_rt_mean"])
        if self._has(feats, ["Age_num", "A2_rt_mean"]):
            feats["Age_A2_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["A2_rt_mean"])
        if self._has(feats, ["Age_num", "A_overall_rt_mean"]):
            feats["Age_overall_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["A_overall_rt_mean"])
        
        # Nonlinear transformations
        if self._has(feats, ["A1_rt_mean"]):
            a1_mean_num = self._ensure_numeric(feats["A1_rt_mean"])
            feats["A1_rt_mean_squared"] = a1_mean_num ** 2
            feats["A1_rt_mean_log"] = np.log1p(a1_mean_num)
        
        if self._has(feats, ["A2_rt_mean"]):
            a2_mean_num = self._ensure_numeric(feats["A2_rt_mean"])
            feats["A2_rt_mean_squared"] = a2_mean_num ** 2
            feats["A2_rt_mean_log"] = np.log1p(a2_mean_num)
        
        if self._has(feats, ["A4_rt_mean"]):
            a4_mean_num = self._ensure_numeric(feats["A4_rt_mean"])
            feats["A4_rt_mean_squared"] = a4_mean_num ** 2
        
        # Response accuracy variance
        if self._has(feats, ["A4_acc_rate", "A5_acc_rate"]):
            feats["A4_A5_acc_diff"] = self._safe_subtract(feats["A4_acc_rate"], feats["A5_acc_rate"])
        
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Feature engineering complete for test A: {feats.shape}")
        return feats
    
    def add_features_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for test type B with domain knowledge.
        
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
        
        # Year-Month index
        if self._has(feats, ["Year", "Month"]):
            feats["YearMonthIndex"] = self._safe_multiply(feats["Year"], 12) + self._ensure_numeric(feats["Month"])
        
        # Basic speed-accuracy tradeoff features
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
        
        # Temporal adaptation features
        if self._has(feats, ["B1_rt_first_half", "B1_rt_second_half"]):
            feats["B1_rt_adaptation"] = self._safe_subtract(feats["B1_rt_second_half"], feats["B1_rt_first_half"])
            feats["B1_rt_learning_rate"] = self._safe_div(
                self._safe_subtract(feats["B1_rt_first_half"], feats["B1_rt_second_half"]),
                feats["B1_rt_first_half"], eps
            )
        
        if self._has(feats, ["B2_rt_first_half", "B2_rt_second_half"]):
            feats["B2_rt_adaptation"] = self._safe_subtract(feats["B2_rt_second_half"], feats["B2_rt_first_half"])
        
        # Extreme performance indicators
        if self._has(feats, ["B1_rt_range", "B1_rt_mean"]):
            feats["B1_rt_extreme_ratio"] = self._safe_div(feats["B1_rt_range"], feats["B1_rt_mean"], eps)
        
        if self._has(feats, ["B2_rt_range", "B2_rt_mean"]):
            feats["B2_rt_extreme_ratio"] = self._safe_div(feats["B2_rt_range"], feats["B2_rt_mean"], eps)
        
        if self._has(feats, ["B3_rt_range", "B3_rt_mean"]):
            feats["B3_rt_extreme_ratio"] = self._safe_div(feats["B3_rt_range"], feats["B3_rt_mean"], eps)
        
        if self._has(feats, ["B4_rt_range", "B4_rt_mean"]):
            feats["B4_rt_extreme_ratio"] = self._safe_div(feats["B4_rt_range"], feats["B4_rt_mean"], eps)
        
        if self._has(feats, ["B5_rt_range", "B5_rt_mean"]):
            feats["B5_rt_extreme_ratio"] = self._safe_div(feats["B5_rt_range"], feats["B5_rt_mean"], eps)
        
        # Cross-test performance comparison
        if self._has(feats, ["B1_rt_mean", "B2_rt_mean"]):
            feats["B1_B2_rt_ratio"] = self._safe_div(feats["B1_rt_mean"], feats["B2_rt_mean"], eps)
            feats["B1_B2_rt_diff"] = self._safe_subtract(feats["B1_rt_mean"], feats["B2_rt_mean"])
        
        if self._has(feats, ["B1_rt_mean", "B3_rt_mean"]):
            feats["B1_B3_rt_ratio"] = self._safe_div(feats["B1_rt_mean"], feats["B3_rt_mean"], eps)
        
        if self._has(feats, ["B3_rt_mean", "B4_rt_mean"]):
            feats["B3_B4_rt_ratio"] = self._safe_div(feats["B3_rt_mean"], feats["B4_rt_mean"], eps)
        
        if self._has(feats, ["B4_rt_mean", "B5_rt_mean"]):
            feats["B4_B5_rt_ratio"] = self._safe_div(feats["B4_rt_mean"], feats["B5_rt_mean"], eps)
        
        # Cross-test consistency comparison
        if self._has(feats, ["B1_rt_consistency", "B2_rt_consistency"]):
            feats["B1_B2_consistency_diff"] = self._safe_subtract(feats["B1_rt_consistency"], feats["B2_rt_consistency"])
        
        if self._has(feats, ["B3_rt_consistency", "B4_rt_consistency"]):
            feats["B3_B4_consistency_diff"] = self._safe_subtract(feats["B3_rt_consistency"], feats["B4_rt_consistency"])
        
        # Task comparison within test
        if self._has(feats, ["B1_acc_task1", "B1_acc_task2"]):
            feats["B1_task_consistency"] = 1.0 - self._safe_subtract(feats["B1_acc_task1"], feats["B1_acc_task2"]).abs()
        
        if self._has(feats, ["B2_acc_task1", "B2_acc_task2"]):
            feats["B2_task_consistency"] = 1.0 - self._safe_subtract(feats["B2_acc_task1"], feats["B2_acc_task2"]).abs()
        
        # Median-based robustness
        if self._has(feats, ["B1_rt_median", "B1_rt_mean"]):
            feats["B1_rt_robustness"] = self._safe_div(feats["B1_rt_median"], feats["B1_rt_mean"], eps)
        
        if self._has(feats, ["B2_rt_median", "B2_rt_mean"]):
            feats["B2_rt_robustness"] = self._safe_div(feats["B2_rt_median"], feats["B2_rt_mean"], eps)
        
        if self._has(feats, ["B3_rt_median", "B3_rt_mean"]):
            feats["B3_rt_robustness"] = self._safe_div(feats["B3_rt_median"], feats["B3_rt_mean"], eps)
        
        # Skewness-based asymmetry
        if self._has(feats, ["B1_rt_skew"]):
            feats["B1_rt_skew_abs"] = self._ensure_numeric(feats["B1_rt_skew"]).abs()
        if self._has(feats, ["B2_rt_skew"]):
            feats["B2_rt_skew_abs"] = self._ensure_numeric(feats["B2_rt_skew"]).abs()
        if self._has(feats, ["B3_rt_skew"]):
            feats["B3_rt_skew_abs"] = self._ensure_numeric(feats["B3_rt_skew"]).abs()
        if self._has(feats, ["B4_rt_skew"]):
            feats["B4_rt_skew_abs"] = self._ensure_numeric(feats["B4_rt_skew"]).abs()
        if self._has(feats, ["B5_rt_skew"]):
            feats["B5_rt_skew_abs"] = self._ensure_numeric(feats["B5_rt_skew"]).abs()
        
        # Overall response quality
        acc_cols = [c for c in feats.columns if 'acc_' in c and c.startswith('B')]
        if len(acc_cols) > 0:
            acc_df = feats[acc_cols].apply(self._ensure_numeric)
            feats["B_overall_acc"] = acc_df.mean(axis=1)
            feats["B_overall_acc_std"] = acc_df.std(axis=1)
            feats["B_overall_acc_min"] = acc_df.min(axis=1)
            feats["B_overall_acc_max"] = acc_df.max(axis=1)
        
        # Overall reaction time profile
        rt_mean_cols = [c for c in feats.columns if c.endswith('_rt_mean') and c.startswith('B')]
        if len(rt_mean_cols) > 0:
            rt_df = feats[rt_mean_cols].apply(self._ensure_numeric)
            feats["B_overall_rt_mean"] = rt_df.mean(axis=1)
            feats["B_overall_rt_std"] = rt_df.std(axis=1)
            feats["B_overall_rt_cv"] = self._safe_div(feats["B_overall_rt_std"], feats["B_overall_rt_mean"], eps)
            feats["B_overall_rt_min"] = rt_df.min(axis=1)
            feats["B_overall_rt_max"] = rt_df.max(axis=1)
        
        # Overall consistency profile
        consistency_cols = [c for c in feats.columns if 'consistency' in c and c.startswith('B')]
        if len(consistency_cols) > 0:
            cons_df = feats[consistency_cols].apply(self._ensure_numeric)
            feats["B_overall_consistency"] = cons_df.mean(axis=1)
            feats["B_overall_consistency_std"] = cons_df.std(axis=1)
            feats["B_overall_consistency_min"] = cons_df.min(axis=1)
        
        # Trend uniformity
        trend_cols = [c for c in feats.columns if 'trend' in c and c.startswith('B')]
        if len(trend_cols) > 0:
            trend_df = feats[trend_cols].apply(self._ensure_numeric)
            feats["B_overall_trend_mean"] = trend_df.mean(axis=1)
            feats["B_overall_trend_std"] = trend_df.std(axis=1)
        
        # Risk score composite (from original)
        parts = []
        for k in ["B4", "B5"]:
            if self._has(feats, [f"{k}_rt_cv"]):
                parts.append(0.25 * self._ensure_numeric(feats[f"{k}_rt_cv"]).fillna(0))
        for k in ["B3", "B4", "B5"]:
            acc = f"{k}_acc_rate"
            if acc in feats:
                parts.append(0.25 * (1 - self._ensure_numeric(feats[acc]).fillna(0)))
        for k in ["B1", "B2"]:
            tcol = f"{k}_speed_acc_tradeoff"
            if tcol in feats:
                parts.append(0.25 * self._ensure_numeric(feats[tcol]).fillna(0))
        if parts:
            feats["RiskScore_B"] = sum(parts)
        
        # Enhanced risk score with temporal features
        risk_parts = []
        if self._has(feats, ["B_overall_rt_cv"]):
            risk_parts.append(0.2 * self._ensure_numeric(feats["B_overall_rt_cv"]).fillna(0))
        if self._has(feats, ["B_overall_acc"]):
            risk_parts.append(0.2 * (1 - self._ensure_numeric(feats["B_overall_acc"]).fillna(0)))
        if self._has(feats, ["B_overall_consistency"]):
            risk_parts.append(0.2 * (1 - self._ensure_numeric(feats["B_overall_consistency"]).fillna(0)))
        if self._has(feats, ["B1_rt_outlier_count"]):
            outlier_norm = self._ensure_numeric(feats["B1_rt_outlier_count"]).fillna(0) / 10.0
            risk_parts.append(0.2 * outlier_norm)
        if self._has(feats, ["B_overall_trend_std"]):
            risk_parts.append(0.2 * self._ensure_numeric(feats["B_overall_trend_std"]).fillna(0))
        if risk_parts:
            feats["RiskScore_B_v2"] = sum(risk_parts)
        
        # Age interaction features
        if self._has(feats, ["Age_num", "B1_rt_mean"]):
            feats["Age_B1_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["B1_rt_mean"])
        if self._has(feats, ["Age_num", "B_overall_rt_mean"]):
            feats["Age_overall_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["B_overall_rt_mean"])
        if self._has(feats, ["Age_num", "B_overall_acc"]):
            feats["Age_overall_acc_interaction"] = self._safe_multiply(feats["Age_num"], feats["B_overall_acc"])
        
        # Nonlinear transformations
        if self._has(feats, ["B1_rt_mean"]):
            b1_mean_num = self._ensure_numeric(feats["B1_rt_mean"])
            feats["B1_rt_mean_squared"] = b1_mean_num ** 2
            feats["B1_rt_mean_log"] = np.log1p(b1_mean_num)
        
        if self._has(feats, ["B2_rt_mean"]):
            b2_mean_num = self._ensure_numeric(feats["B2_rt_mean"])
            feats["B2_rt_mean_squared"] = b2_mean_num ** 2
            feats["B2_rt_mean_log"] = np.log1p(b2_mean_num)
        
        if self._has(feats, ["B_overall_rt_mean"]):
            b_overall_mean_num = self._ensure_numeric(feats["B_overall_rt_mean"])
            feats["B_overall_rt_mean_log"] = np.log1p(b_overall_mean_num)
        
        # Outlier propensity
        if self._has(feats, ["B1_rt_outlier_count"]):
            feats["B1_outlier_rate"] = self._ensure_numeric(feats["B1_rt_outlier_count"]) / 20.0
        
        if self._has(feats, ["B3_rt_outlier_count"]):
            feats["B3_outlier_rate"] = self._ensure_numeric(feats["B3_rt_outlier_count"]) / 20.0
        
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