# src/utils.py
# Common utility functions

import logging
import sys
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm

def setup_logging(level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('traffic_risk_prediction')
    logger.info("Logging initialized")
    
    return logger

def validate_predictions(predictions: np.ndarray, expected_length: int) -> None:
    """
    Validate prediction output.
    
    Args:
        predictions: Array of predictions
        expected_length: Expected number of predictions
        
    Raises:
        ValueError: If validation fails
    """
    # Check length
    if len(predictions) != expected_length:
        raise ValueError(
            f"Prediction length mismatch: expected {expected_length}, "
            f"got {len(predictions)}"
        )
    
    # Check for NaN/Inf
    if np.isnan(predictions).any():
        raise ValueError("Predictions contain NaN values")
    
    if np.isinf(predictions).any():
        raise ValueError("Predictions contain infinite values")
    
    # Check range
    if not np.all((predictions >= 0) & (predictions <= 1)):
        raise ValueError("Predictions must be in range [0, 1]")
    
    logging.info("Prediction validation passed")

def calculate_memory_usage(df: pd.DataFrame) -> str:
    """
    Calculate memory usage of DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        str: Memory usage summary
    """
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    return f"{memory_mb:.2f} MB"

def convert_age(val):
    """
    Convert age string format to numeric.
    
    Args:
        val: Age string (e.g., '30a' or '30b')
        
    Returns:
        float: Numeric age value
    """
    if pd.isna(val):
        return np.nan
    try:
        base = int(str(val)[:-1])
        return base if str(val)[-1] == "a" else base + 5
    except:
        return np.nan

def split_testdate(val):
    """
    Split test date into year and month.
    
    Args:
        val: Test date in YYYYMM format
        
    Returns:
        tuple: (year, month)
    """
    try:
        v = int(val)
        return v // 100, v % 100
    except:
        return np.nan, np.nan

def seq_mean(series):
    """
    Calculate mean of comma-separated sequence values.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Mean values
    """
    # Convert to string to handle categorical type
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )

def seq_std(series):
    """
    Calculate standard deviation of comma-separated sequence values.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Standard deviation values
    """
    # Convert to string to handle categorical type
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )

def seq_rate(series, target="1"):
    """
    Calculate rate of target value in comma-separated sequences.
    
    Args:
        series: Pandas Series with comma-separated values
        target: Target value to count
        
    Returns:
        pd.Series: Rate values
    """
    # Convert to string to handle categorical type
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: str(x).split(",").count(target) / len(x.split(",")) if x else np.nan
    )

def masked_mean_from_csv_series(cond_series, val_series, mask_val):
    """
    Calculate masked mean from CSV series.
    
    Args:
        cond_series: Condition series (comma-separated)
        val_series: Value series (comma-separated)
        mask_val: Value to mask on
        
    Returns:
        pd.Series: Masked mean values
    """
    # Convert to string to handle categorical type
    cond_str = cond_series.astype(str).replace('nan', '')
    val_str = val_series.astype(str).replace('nan', '')
    
    cond_df = cond_str.str.split(",", expand=True).replace("", np.nan)
    val_df = val_str.str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)

def masked_mean_in_set_series(cond_series, val_series, mask_set):
    """
    Calculate masked mean for values in a set.
    
    Args:
        cond_series: Condition series (comma-separated)
        val_series: Value series (comma-separated)
        mask_set: Set of values to mask on
        
    Returns:
        pd.Series: Masked mean values
    """
    # Convert to string to handle categorical type
    cond_str = cond_series.astype(str).replace('nan', '')
    val_str = val_series.astype(str).replace('nan', '')
    
    cond_df = cond_str.str.split(",", expand=True).replace("", np.nan)
    val_df = val_str.str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr = val_df.to_numpy(dtype=float)
    mask = np.isin(cond_arr, list(mask_set))
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)