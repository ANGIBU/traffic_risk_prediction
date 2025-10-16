# src/preprocessor.py
# Data preprocessing pipeline

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple
from .utils import (
    convert_age, split_testdate, seq_mean, seq_std, seq_rate,
    masked_mean_from_csv_series, masked_mean_in_set_series
)

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Data preprocessing pipeline.
    
    Handles missing values, outliers, and type-specific preprocessing
    for test types A and B following baseline methodology.
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.preprocessing_config = config.preprocessing
        self.eps = self.preprocessing_config['eps']
        
    def preprocess_a(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess test type A data.
        
        Args:
            df: Type A data
            
        Returns:
            pd.DataFrame: Preprocessed type A data
        """
        if len(df) == 0:
            return df
        
        df = df.copy()
        logger.info("Step 1: Age, TestDate features")
        
        # Age and date features
        df["Age_num"] = df["Age"].map(convert_age)
        ym = df["TestDate"].map(split_testdate)
        df["Year"] = [y for y, m in ym]
        df["Month"] = [m for y, m in ym]
        
        feats = pd.DataFrame(index=df.index)
        
        logger.info("Step 2: A1 feature generation")
        feats["A1_resp_rate"] = seq_rate(df["A1-3"], "1")
        feats["A1_rt_mean"] = seq_mean(df["A1-4"])
        feats["A1_rt_std"] = seq_std(df["A1-4"])
        feats["A1_rt_left"] = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 1)
        feats["A1_rt_right"] = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 2)
        feats["A1_rt_side_diff"] = feats["A1_rt_left"] - feats["A1_rt_right"]
        feats["A1_rt_slow"] = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 1)
        feats["A1_rt_fast"] = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 3)
        feats["A1_rt_speed_diff"] = feats["A1_rt_slow"] - feats["A1_rt_fast"]
        
        logger.info("Step 3: A2 feature generation")
        feats["A2_resp_rate"] = seq_rate(df["A2-3"], "1")
        feats["A2_rt_mean"] = seq_mean(df["A2-4"])
        feats["A2_rt_std"] = seq_std(df["A2-4"])
        feats["A2_rt_cond1_diff"] = masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 1) - \
                                    masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 3)
        feats["A2_rt_cond2_diff"] = masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 1) - \
                                    masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 3)
        
        logger.info("Step 4: A3 feature generation")
        s = df["A3-5"].fillna("")
        total = s.apply(lambda x: len(x.split(",")) if x else 0)
        valid = s.apply(lambda x: sum(v in {"1", "2"} for v in x.split(",")) if x else 0)
        invalid = s.apply(lambda x: sum(v in {"3", "4"} for v in x.split(",")) if x else 0)
        correct = s.apply(lambda x: sum(v in {"1", "3"} for v in x.split(",")) if x else 0)
        feats["A3_valid_ratio"] = (valid / total).replace([np.inf, -np.inf], np.nan)
        feats["A3_invalid_ratio"] = (invalid / total).replace([np.inf, -np.inf], np.nan)
        feats["A3_correct_ratio"] = (correct / total).replace([np.inf, -np.inf], np.nan)
        
        feats["A3_resp2_rate"] = seq_rate(df["A3-6"], "1")
        feats["A3_rt_mean"] = seq_mean(df["A3-7"])
        feats["A3_rt_std"] = seq_std(df["A3-7"])
        feats["A3_rt_size_diff"] = masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 1) - \
                                   masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2)
        feats["A3_rt_side_diff"] = masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 1) - \
                                   masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2)
        
        logger.info("Step 5: A4 feature generation")
        feats["A4_acc_rate"] = seq_rate(df["A4-3"], "1")
        feats["A4_resp2_rate"] = seq_rate(df["A4-4"], "1")
        feats["A4_rt_mean"] = seq_mean(df["A4-5"])
        feats["A4_rt_std"] = seq_std(df["A4-5"])
        feats["A4_stroop_diff"] = masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 2) - \
                                  masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)
        feats["A4_rt_color_diff"] = masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 1) - \
                                    masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2)
        
        logger.info("Step 6: A5 feature generation")
        feats["A5_acc_rate"] = seq_rate(df["A5-2"], "1")
        feats["A5_resp2_rate"] = seq_rate(df["A5-3"], "1")
        feats["A5_acc_nonchange"] = masked_mean_from_csv_series(df["A5-1"], df["A5-2"], 1)
        feats["A5_acc_change"] = masked_mean_in_set_series(df["A5-1"], df["A5-2"], {2, 3, 4})
        
        logger.info("Step 7: Drop sequence columns and concat")
        seq_cols = [
            "A1-1", "A1-2", "A1-3", "A1-4",
            "A2-1", "A2-2", "A2-3", "A2-4",
            "A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7",
            "A4-1", "A4-2", "A4-3", "A4-4", "A4-5",
            "A5-1", "A5-2", "A5-3"
        ]
        logger.info("A test data preprocessing complete")
        
        out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out
    
    def preprocess_b(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess test type B data.
        
        Args:
            df: Type B data
            
        Returns:
            pd.DataFrame: Preprocessed type B data
        """
        if len(df) == 0:
            return df
        
        df = df.copy()
        logger.info("Step 1: Age, TestDate features")
        
        # Age and date features
        df["Age_num"] = df["Age"].map(convert_age)
        ym = df["TestDate"].map(split_testdate)
        df["Year"] = [y for y, m in ym]
        df["Month"] = [m for y, m in ym]
        
        feats = pd.DataFrame(index=df.index)
        
        logger.info("Step 2: B1 feature generation")
        feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1")
        feats["B1_rt_mean"] = seq_mean(df["B1-2"])
        feats["B1_rt_std"] = seq_std(df["B1-2"])
        feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1")
        
        logger.info("Step 3: B2 feature generation")
        feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1")
        feats["B2_rt_mean"] = seq_mean(df["B2-2"])
        feats["B2_rt_std"] = seq_std(df["B2-2"])
        feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1")
        
        logger.info("Step 4: B3 feature generation")
        feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
        feats["B3_rt_mean"] = seq_mean(df["B3-2"])
        feats["B3_rt_std"] = seq_std(df["B3-2"])
        
        logger.info("Step 5: B4 feature generation")
        feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
        feats["B4_rt_mean"] = seq_mean(df["B4-2"])
        feats["B4_rt_std"] = seq_std(df["B4-2"])
        
        logger.info("Step 6: B5 feature generation")
        feats["B5_acc_rate"] = seq_rate(df["B5-1"], "1")
        feats["B5_rt_mean"] = seq_mean(df["B5-2"])
        feats["B5_rt_std"] = seq_std(df["B5-2"])
        
        logger.info("Step 7: B6~B8 feature generation")
        feats["B6_acc_rate"] = seq_rate(df["B6"], "1")
        feats["B7_acc_rate"] = seq_rate(df["B7"], "1")
        feats["B8_acc_rate"] = seq_rate(df["B8"], "1")
        
        logger.info("Step 8: Drop sequence columns and concat")
        seq_cols = [
            "B1-1", "B1-2", "B1-3",
            "B2-1", "B2-2", "B2-3",
            "B3-1", "B3-2",
            "B4-1", "B4-2",
            "B5-1", "B5-2",
            "B6", "B7", "B8"
        ]
        logger.info("B test data preprocessing complete")
        
        out = pd.concat([df.drop(columns=seq_cols, errors="ignore"), feats], axis=1)
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        return out
