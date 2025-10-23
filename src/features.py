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
        Add derived features for test type A with domain knowledge and Type A specific enhancements.
        
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
        
        # Coefficient of variation for reaction time (only key tests)
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
        
        # Temporal adaptation features (first vs second half)
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
        
        # Cross-test performance comparison (key pairs only)
        if self._has(feats, ["A1_rt_mean", "A4_rt_mean"]):
            feats["A1_A4_rt_ratio"] = self._safe_div(feats["A1_rt_mean"], feats["A4_rt_mean"], eps)
        
        if self._has(feats, ["A3_rt_mean", "A4_rt_mean"]):
            feats["A3_A4_rt_ratio"] = self._safe_div(feats["A3_rt_mean"], feats["A4_rt_mean"], eps)
        
        # Median-based robustness (key measure)
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
        
        # Age interaction features with binning
        if self._has(feats, ["Age_num", "A_overall_rt_mean"]):
            feats["Age_overall_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["A_overall_rt_mean"])
        
        if self._has(feats, ["Age_num", "A4_rt_mean"]):
            feats["Age_A4_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["A4_rt_mean"])
        
        if self._has(feats, ["Age_num", "A_overall_consistency"]):
            feats["Age_consistency_interaction"] = self._safe_multiply(feats["Age_num"], feats["A_overall_consistency"])
        
        # ========== NEW TYPE A SPECIFIC FEATURES ==========
        
        # 1. A1-A2-A3 Sequential Pattern Analysis
        if self._has(feats, ["A1_rt_mean", "A2_rt_mean", "A3_rt_mean"]):
            a1_rt = self._ensure_numeric(feats["A1_rt_mean"])
            a2_rt = self._ensure_numeric(feats["A2_rt_mean"])
            a3_rt = self._ensure_numeric(feats["A3_rt_mean"])
            
            feats["A123_rt_progression"] = (a2_rt - a1_rt) + (a3_rt - a2_rt)
            feats["A123_rt_acceleration"] = (a3_rt - a2_rt) - (a2_rt - a1_rt)
            feats["A123_rt_consistency"] = 1.0 - pd.DataFrame({
                'a1': a1_rt, 'a2': a2_rt, 'a3': a3_rt
            }).std(axis=1) / (pd.DataFrame({
                'a1': a1_rt, 'a2': a2_rt, 'a3': a3_rt
            }).mean(axis=1) + eps)
        
        # 2. Stroop Effect Detailed Analysis (A4)
        if self._has(feats, ["A4_stroop_diff", "A4_rt_mean"]):
            stroop_diff = self._ensure_numeric(feats["A4_stroop_diff"])
            a4_rt = self._ensure_numeric(feats["A4_rt_mean"])
            
            feats["A4_stroop_effect_ratio"] = self._safe_div(stroop_diff, a4_rt, eps)
            feats["A4_stroop_interference"] = stroop_diff.abs()
        
        # 3. Attention Shift Pattern (A1-A2 comparison)
        if self._has(feats, ["A1_resp_rate", "A2_resp_rate"]):
            a1_resp = self._ensure_numeric(feats["A1_resp_rate"])
            a2_resp = self._ensure_numeric(feats["A2_resp_rate"])
            
            feats["A12_attention_shift"] = a2_resp - a1_resp
            feats["A12_attention_stability"] = 1.0 - (a2_resp - a1_resp).abs()
        
        # 4. Visual Search Efficiency (A3)
        if self._has(feats, ["A3_rt_mean", "A3_resp2_rate"]):
            a3_rt = self._ensure_numeric(feats["A3_rt_mean"])
            a3_resp = self._ensure_numeric(feats["A3_resp2_rate"])
            
            feats["A3_search_efficiency"] = self._safe_div(a3_resp, a3_rt, eps)
        
        if self._has(feats, ["A3_valid_ratio", "A3_invalid_ratio"]):
            valid = self._ensure_numeric(feats["A3_valid_ratio"])
            invalid = self._ensure_numeric(feats["A3_invalid_ratio"])
            
            feats["A3_discrimination_ability"] = valid - invalid
            feats["A3_error_rate"] = invalid
        
        # 5. Change Detection Sensitivity (A5)
        if self._has(feats, ["A5_acc_nonchange", "A5_acc_change"]):
            nonchange = self._ensure_numeric(feats["A5_acc_nonchange"])
            change = self._ensure_numeric(feats["A5_acc_change"])
            
            feats["A5_detection_bias"] = nonchange - change
            feats["A5_detection_balance"] = 1.0 - (nonchange - change).abs()
        
        # 6. Consistency across all tests
        if self._has(feats, ["A1_rt_consistency", "A2_rt_consistency", "A4_rt_consistency"]):
            a1_cons = self._ensure_numeric(feats["A1_rt_consistency"])
            a2_cons = self._ensure_numeric(feats["A2_rt_consistency"])
            a4_cons = self._ensure_numeric(feats["A4_rt_consistency"])
            
            feats["A_consistency_stability"] = 1.0 - pd.DataFrame({
                'a1': a1_cons, 'a2': a2_cons, 'a4': a4_cons
            }).std(axis=1)
        
        # 7. Outlier tendency
        if self._has(feats, ["A1_rt_outlier_count", "A3_rt_outlier_count", "A4_rt_outlier_count"]):
            a1_out = self._ensure_numeric(feats["A1_rt_outlier_count"])
            a3_out = self._ensure_numeric(feats["A3_rt_outlier_count"])
            a4_out = self._ensure_numeric(feats["A4_rt_outlier_count"])
            
            feats["A_total_outliers"] = a1_out + a3_out + a4_out
            feats["A_outlier_consistency"] = 1.0 - pd.DataFrame({
                'a1': a1_out, 'a3': a3_out, 'a4': a4_out
            }).std(axis=1) / (pd.DataFrame({
                'a1': a1_out, 'a3': a3_out, 'a4': a4_out
            }).mean(axis=1) + eps)
        
        # 8. IQR-based stability
        if self._has(feats, ["A1_rt_iqr", "A3_rt_iqr", "A4_rt_iqr"]):
            a1_iqr = self._ensure_numeric(feats["A1_rt_iqr"])
            a3_iqr = self._ensure_numeric(feats["A3_rt_iqr"])
            a4_iqr = self._ensure_numeric(feats["A4_rt_iqr"])
            
            feats["A_iqr_mean"] = (a1_iqr + a3_iqr + a4_iqr) / 3.0
            feats["A_iqr_stability"] = 1.0 - pd.DataFrame({
                'a1': a1_iqr, 'a3': a3_iqr, 'a4': a4_iqr
            }).std(axis=1) / (feats["A_iqr_mean"] + eps)
        
        # 9. Speed change pattern
        if self._has(feats, ["A1_rt_change_rate", "A2_rt_change_rate", "A4_rt_change_rate"]):
            a1_change = self._ensure_numeric(feats["A1_rt_change_rate"])
            a2_change = self._ensure_numeric(feats["A2_rt_change_rate"])
            a4_change = self._ensure_numeric(feats["A4_rt_change_rate"])
            
            feats["A_speed_learning"] = (a1_change + a2_change + a4_change) / 3.0
            feats["A_speed_decline"] = feats["A_speed_learning"].apply(lambda x: max(0, -x))
        
        # 10. Composite risk indicators for Type A
        risk_components = []
        if self._has(feats, ["A_overall_rt_std", "A_overall_rt_mean"]):
            cv = self._safe_div(feats["A_overall_rt_std"], feats["A_overall_rt_mean"], eps)
            risk_components.append(0.3 * self._ensure_numeric(cv).fillna(0))
        if self._has(feats, ["A_overall_acc"]):
            risk_components.append(0.3 * (1 - self._ensure_numeric(feats["A_overall_acc"]).fillna(0)))
        if self._has(feats, ["A_overall_consistency"]):
            risk_components.append(0.2 * (1 - self._ensure_numeric(feats["A_overall_consistency"]).fillna(0)))
        if self._has(feats, ["A_total_outliers"]):
            outliers_norm = self._ensure_numeric(feats["A_total_outliers"]).fillna(0) / 10.0
            risk_components.append(0.2 * outliers_norm.clip(0, 1))
        
        if risk_components:
            feats["RiskScore_A"] = sum(risk_components)
        
        # ========== END OF NEW TYPE A FEATURES ==========
        
        feats.replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f"Feature engineering complete for test A: {feats.shape}")
        return feats
    
    def add_features_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for test type B with domain knowledge and Type B specific patterns.
        
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
        
        # Coefficient of variation for reaction time (key tests only)
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
        
        # Cross-test performance comparison (key pairs)
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
        
        # Sequential test pattern (B1→B2→B3)
        if self._has(feats, ["B1_rt_mean", "B2_rt_mean", "B3_rt_mean"]):
            b1_num = self._ensure_numeric(feats["B1_rt_mean"])
            b2_num = self._ensure_numeric(feats["B2_rt_mean"])
            b3_num = self._ensure_numeric(feats["B3_rt_mean"])
            feats["B123_rt_progression"] = (b2_num - b1_num) + (b3_num - b2_num)
            feats["B123_rt_variability"] = pd.DataFrame({
                'b1': b1_num, 'b2': b2_num, 'b3': b3_num
            }).std(axis=1)
        
        # Sequential test pattern (B3→B4→B5)
        if self._has(feats, ["B3_rt_mean", "B4_rt_mean", "B5_rt_mean"]):
            b3_num = self._ensure_numeric(feats["B3_rt_mean"])
            b4_num = self._ensure_numeric(feats["B4_rt_mean"])
            b5_num = self._ensure_numeric(feats["B5_rt_mean"])
            feats["B345_rt_progression"] = (b4_num - b3_num) + (b5_num - b4_num)
            feats["B345_rt_variability"] = pd.DataFrame({
                'b3': b3_num, 'b4': b4_num, 'b5': b5_num
            }).std(axis=1)
        
        # Accuracy progression pattern
        if self._has(feats, ["B3_acc_rate", "B4_acc_rate", "B5_acc_rate"]):
            b3_acc = self._ensure_numeric(feats["B3_acc_rate"])
            b4_acc = self._ensure_numeric(feats["B4_acc_rate"])
            b5_acc = self._ensure_numeric(feats["B5_acc_rate"])
            feats["B345_acc_decline"] = b3_acc - b5_acc
            feats["B345_acc_stability"] = 1.0 - pd.DataFrame({
                'b3': b3_acc, 'b4': b4_acc, 'b5': b5_acc
            }).std(axis=1)
        
        # Attention consistency across B6-B7-B8
        if self._has(feats, ["B6_acc_rate", "B7_acc_rate", "B8_acc_rate"]):
            b6_acc = self._ensure_numeric(feats["B6_acc_rate"])
            b7_acc = self._ensure_numeric(feats["B7_acc_rate"])
            b8_acc = self._ensure_numeric(feats["B8_acc_rate"])
            feats["B678_acc_consistency"] = 1.0 - pd.DataFrame({
                'b6': b6_acc, 'b7': b7_acc, 'b8': b8_acc
            }).std(axis=1)
            feats["B678_acc_min"] = pd.DataFrame({
                'b6': b6_acc, 'b7': b7_acc, 'b8': b8_acc
            }).min(axis=1)
        
        # Median-based robustness (key measure)
        if self._has(feats, ["B3_rt_median", "B3_rt_mean"]):
            feats["B3_rt_robustness"] = self._safe_div(feats["B3_rt_median"], feats["B3_rt_mean"], eps)
        
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
        
        # ========== NEW TYPE B SPECIFIC FEATURES ==========
        
        # 1. B1->B2->B3 Full Sequential Flow Analysis
        if self._has(feats, ["B1_acc_task1", "B2_acc_task1", "B3_acc_rate"]):
            b1_acc = self._ensure_numeric(feats["B1_acc_task1"])
            b2_acc = self._ensure_numeric(feats["B2_acc_task1"])
            b3_acc = self._ensure_numeric(feats["B3_acc_rate"])
            
            feats["B123_acc_monotonic_decline"] = ((b1_acc >= b2_acc) & (b2_acc >= b3_acc)).astype(float)
            feats["B123_acc_total_change"] = b1_acc - b3_acc
            feats["B123_acc_change_rate"] = self._safe_div(b1_acc - b3_acc, b1_acc, eps)
        
        if self._has(feats, ["B1_rt_mean", "B2_rt_mean", "B3_rt_mean"]):
            b1_rt = self._ensure_numeric(feats["B1_rt_mean"])
            b2_rt = self._ensure_numeric(feats["B2_rt_mean"])
            b3_rt = self._ensure_numeric(feats["B3_rt_mean"])
            
            feats["B123_rt_monotonic_increase"] = ((b1_rt <= b2_rt) & (b2_rt <= b3_rt)).astype(float)
            feats["B123_rt_acceleration"] = (b3_rt - b2_rt) - (b2_rt - b1_rt)
        
        # 2. B3->B4->B5 Detailed Change Rate Analysis
        if self._has(feats, ["B3_acc_rate", "B4_acc_rate", "B5_acc_rate"]):
            b3_acc = self._ensure_numeric(feats["B3_acc_rate"])
            b4_acc = self._ensure_numeric(feats["B4_acc_rate"])
            b5_acc = self._ensure_numeric(feats["B5_acc_rate"])
            
            feats["B34_acc_change_rate"] = self._safe_div(b4_acc - b3_acc, b3_acc, eps)
            feats["B45_acc_change_rate"] = self._safe_div(b5_acc - b4_acc, b4_acc, eps)
            feats["B345_acc_change_acceleration"] = feats["B45_acc_change_rate"] - feats["B34_acc_change_rate"]
        
        if self._has(feats, ["B3_rt_mean", "B4_rt_mean", "B5_rt_mean"]):
            b3_rt = self._ensure_numeric(feats["B3_rt_mean"])
            b4_rt = self._ensure_numeric(feats["B4_rt_mean"])
            b5_rt = self._ensure_numeric(feats["B5_rt_mean"])
            
            feats["B34_rt_change_rate"] = self._safe_div(b4_rt - b3_rt, b3_rt, eps)
            feats["B45_rt_change_rate"] = self._safe_div(b5_rt - b4_rt, b4_rt, eps)
            feats["B345_rt_change_stability"] = 1.0 - (feats["B34_rt_change_rate"] - feats["B45_rt_change_rate"]).abs()
        
        # 3. Task Switching Cost Analysis
        if self._has(feats, ["B1_acc_task1", "B1_acc_task2"]):
            b1_t1 = self._ensure_numeric(feats["B1_acc_task1"])
            b1_t2 = self._ensure_numeric(feats["B1_acc_task2"])
            feats["B1_task_switch_cost"] = (b1_t1 - b1_t2).abs()
            feats["B1_task_switch_direction"] = (b1_t1 - b1_t2)
        
        if self._has(feats, ["B2_acc_task1", "B2_acc_task2"]):
            b2_t1 = self._ensure_numeric(feats["B2_acc_task1"])
            b2_t2 = self._ensure_numeric(feats["B2_acc_task2"])
            feats["B2_task_switch_cost"] = (b2_t1 - b2_t2).abs()
            feats["B2_task_switch_direction"] = (b2_t1 - b2_t2)
        
        if self._has(feats, ["B1_task_switch_cost", "B2_task_switch_cost"]):
            b1_switch = self._ensure_numeric(feats["B1_task_switch_cost"])
            b2_switch = self._ensure_numeric(feats["B2_task_switch_cost"])
            feats["B12_task_switch_change"] = b2_switch - b1_switch
            feats["B12_task_switch_stability"] = 1.0 - (b1_switch - b2_switch).abs()
        
        # 4. Attention Maintenance Pattern (B6-B7-B8)
        if self._has(feats, ["B6_acc_rate", "B7_acc_rate", "B8_acc_rate"]):
            b6_acc = self._ensure_numeric(feats["B6_acc_rate"])
            b7_acc = self._ensure_numeric(feats["B7_acc_rate"])
            b8_acc = self._ensure_numeric(feats["B8_acc_rate"])
            
            feats["B67_acc_change"] = b7_acc - b6_acc
            feats["B78_acc_change"] = b8_acc - b7_acc
            feats["B678_acc_change_consistency"] = 1.0 - (feats["B67_acc_change"] - feats["B78_acc_change"]).abs()
            
            feats["B678_acc_range"] = b6_acc.combine(b7_acc, max).combine(b8_acc, max) - \
                                      b6_acc.combine(b7_acc, min).combine(b8_acc, min)
            
            feats["B678_monotonic_decline"] = ((b6_acc >= b7_acc) & (b7_acc >= b8_acc)).astype(float)
        
        if self._has(feats, ["B6_acc_consistency", "B7_acc_consistency", "B8_acc_consistency"]):
            b6_cons = self._ensure_numeric(feats["B6_acc_consistency"])
            b7_cons = self._ensure_numeric(feats["B7_acc_consistency"])
            b8_cons = self._ensure_numeric(feats["B8_acc_consistency"])
            
            feats["B678_consistency_mean"] = (b6_cons + b7_cons + b8_cons) / 3.0
            feats["B678_consistency_decline"] = b6_cons - b8_cons
        
        # 5. Composite Risk Indicators
        parts = []
        if self._has(feats, ["B_overall_rt_std", "B_overall_rt_mean"]):
            cv = self._safe_div(feats["B_overall_rt_std"], feats["B_overall_rt_mean"], eps)
            parts.append(0.25 * self._ensure_numeric(cv).fillna(0))
        if self._has(feats, ["B_overall_acc"]):
            parts.append(0.25 * (1 - self._ensure_numeric(feats["B_overall_acc"]).fillna(0)))
        if self._has(feats, ["B_overall_consistency"]):
            parts.append(0.25 * (1 - self._ensure_numeric(feats["B_overall_consistency"]).fillna(0)))
        if self._has(feats, ["B_overall_acc_min"]):
            parts.append(0.25 * (1 - self._ensure_numeric(feats["B_overall_acc_min"]).fillna(0)))
        if parts:
            feats["RiskScore_B"] = sum(parts)
        
        # Sequential risk indicator
        if self._has(feats, ["B123_acc_total_change", "B345_acc_decline"]):
            early_decline = self._ensure_numeric(feats["B123_acc_total_change"]).fillna(0)
            late_decline = self._ensure_numeric(feats["B345_acc_decline"]).fillna(0)
            feats["Sequential_decline_risk"] = (early_decline + late_decline) / 2.0
        
        # Attention risk indicator
        if self._has(feats, ["B678_acc_min", "B678_acc_consistency"]):
            attention_min = self._ensure_numeric(feats["B678_acc_min"]).fillna(1.0)
            attention_cons = self._ensure_numeric(feats["B678_acc_consistency"]).fillna(1.0)
            feats["Attention_risk"] = (1 - attention_min) * 0.6 + (1 - attention_cons) * 0.4
        
        # Task switching risk indicator
        if self._has(feats, ["B1_task_switch_cost", "B2_task_switch_cost"]):
            b1_switch = self._ensure_numeric(feats["B1_task_switch_cost"]).fillna(0)
            b2_switch = self._ensure_numeric(feats["B2_task_switch_cost"]).fillna(0)
            feats["Task_switch_risk"] = (b1_switch + b2_switch) / 2.0
        
        # Comprehensive Type B risk score
        risk_components = []
        if "Sequential_decline_risk" in feats.columns:
            risk_components.append(0.3 * self._ensure_numeric(feats["Sequential_decline_risk"]).fillna(0))
        if "Attention_risk" in feats.columns:
            risk_components.append(0.3 * self._ensure_numeric(feats["Attention_risk"]).fillna(0))
        if "Task_switch_risk" in feats.columns:
            risk_components.append(0.2 * self._ensure_numeric(feats["Task_switch_risk"]).fillna(0))
        if "RiskScore_B" in feats.columns:
            risk_components.append(0.2 * self._ensure_numeric(feats["RiskScore_B"]).fillna(0))
        
        if risk_components:
            feats["TypeB_comprehensive_risk"] = sum(risk_components)
        
        # ========== END OF NEW FEATURES ==========
        
        # Age interaction features
        if self._has(feats, ["Age_num", "B_overall_rt_mean"]):
            feats["Age_overall_rt_interaction"] = self._safe_multiply(feats["Age_num"], feats["B_overall_rt_mean"])
        
        if self._has(feats, ["Age_num", "B_overall_acc"]):
            feats["Age_overall_acc_interaction"] = self._safe_multiply(feats["Age_num"], feats["B_overall_acc"])
        
        if self._has(feats, ["Age_num", "B_overall_consistency"]):
            feats["Age_consistency_interaction"] = self._safe_multiply(feats["Age_num"], feats["B_overall_consistency"])
        
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