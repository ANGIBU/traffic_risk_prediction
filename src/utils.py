# src/utils.py
# Common utility functions

import logging
import sys
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy import stats

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
    if len(predictions) != expected_length:
        raise ValueError(
            f"Prediction length mismatch: expected {expected_length}, "
            f"got {len(predictions)}"
        )
    
    if np.isnan(predictions).any():
        raise ValueError("Predictions contain NaN values")
    
    if np.isinf(predictions).any():
        raise ValueError("Predictions contain infinite values")
    
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

# New functions for temporal pattern extraction

def seq_trend(series):
    """
    Calculate linear trend (slope) of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Trend slopes
    """
    def calc_slope(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 2:
            return np.nan
        x_vals = np.arange(len(arr))
        try:
            slope, _ = np.polyfit(x_vals, arr, 1)
            return slope
        except:
            return np.nan
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_slope)

def seq_range(series):
    """
    Calculate range (max - min) of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Range values
    """
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: (np.fromstring(x, sep=",").max() - np.fromstring(x, sep=",").min()) if x else np.nan
    )

def seq_min(series):
    """
    Calculate minimum of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Minimum values
    """
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: np.fromstring(x, sep=",").min() if x else np.nan
    )

def seq_max(series):
    """
    Calculate maximum of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Maximum values
    """
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: np.fromstring(x, sep=",").max() if x else np.nan
    )

def seq_median(series):
    """
    Calculate median of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Median values
    """
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: np.median(np.fromstring(x, sep=",")) if x else np.nan
    )

def seq_q25(series):
    """
    Calculate 25th percentile of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Q25 values
    """
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: np.percentile(np.fromstring(x, sep=","), 25) if x else np.nan
    )

def seq_q75(series):
    """
    Calculate 75th percentile of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Q75 values
    """
    series_str = series.astype(str)
    return series_str.replace('nan', '').apply(
        lambda x: np.percentile(np.fromstring(x, sep=","), 75) if x else np.nan
    )

def seq_skew(series):
    """
    Calculate skewness of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Skewness values
    """
    def calc_skew(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 3:
            return np.nan
        return stats.skew(arr, nan_policy='omit')
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_skew)

def seq_kurtosis(series):
    """
    Calculate kurtosis of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Kurtosis values
    """
    def calc_kurt(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 4:
            return np.nan
        return stats.kurtosis(arr, nan_policy='omit')
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_kurt)

def seq_iqr(series):
    """
    Calculate interquartile range of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: IQR values
    """
    def calc_iqr(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 2:
            return np.nan
        return np.percentile(arr, 75) - np.percentile(arr, 25)
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_iqr)

def seq_change_rate(series):
    """
    Calculate rate of change (last - first) / first.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Change rate values
    """
    def calc_change(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 2 or arr[0] == 0:
            return np.nan
        return (arr[-1] - arr[0]) / arr[0]
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_change)

def seq_first_half_mean(series):
    """
    Calculate mean of first half of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: First half mean values
    """
    def calc_first_half(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 2:
            return np.nan
        mid = len(arr) // 2
        return arr[:mid].mean()
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_first_half)

def seq_second_half_mean(series):
    """
    Calculate mean of second half of sequence.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Second half mean values
    """
    def calc_second_half(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 2:
            return np.nan
        mid = len(arr) // 2
        return arr[mid:].mean()
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_second_half)

def seq_consistency(series):
    """
    Calculate consistency as (1 - CV) where CV is coefficient of variation.
    Higher value means more consistent performance.
    
    Args:
        series: Pandas Series with comma-separated values
        
    Returns:
        pd.Series: Consistency scores
    """
    def calc_consistency(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 2 or arr.mean() == 0:
            return np.nan
        cv = arr.std() / arr.mean()
        return 1.0 / (1.0 + cv)
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(calc_consistency)

def seq_outlier_count(series, threshold=2.0):
    """
    Count number of outliers beyond threshold standard deviations.
    
    Args:
        series: Pandas Series with comma-separated values
        threshold: Number of std deviations for outlier threshold
        
    Returns:
        pd.Series: Outlier counts
    """
    def count_outliers(x):
        if not x:
            return np.nan
        arr = np.fromstring(x, sep=",")
        if len(arr) < 3:
            return 0
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0
        z_scores = np.abs((arr - mean) / std)
        return np.sum(z_scores > threshold)
    
    series_str = series.astype(str).replace('nan', '')
    return series_str.apply(count_outliers)

def masked_std_from_csv_series(cond_series, val_series, mask_val):
    """
    Calculate masked standard deviation from CSV series.
    
    Args:
        cond_series: Condition series (comma-separated)
        val_series: Value series (comma-separated)
        mask_val: Value to mask on
        
    Returns:
        pd.Series: Masked std values
    """
    cond_str = cond_series.astype(str).replace('nan', '')
    val_str = val_series.astype(str).replace('nan', '')
    
    cond_df = cond_str.str.split(",", expand=True).replace("", np.nan)
    val_df = val_str.str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    
    def calc_std_row(row_mask, row_vals):
        masked_vals = row_vals[row_mask]
        if len(masked_vals) < 2:
            return np.nan
        return np.nanstd(masked_vals)
    
    result = []
    for i in range(len(cond_arr)):
        result.append(calc_std_row(mask[i], val_arr[i]))
    
    return pd.Series(result, index=cond_series.index)

def masked_max_from_csv_series(cond_series, val_series, mask_val):
    """
    Calculate masked maximum from CSV series.
    
    Args:
        cond_series: Condition series (comma-separated)
        val_series: Value series (comma-separated)
        mask_val: Value to mask on
        
    Returns:
        pd.Series: Masked max values
    """
    cond_str = cond_series.astype(str).replace('nan', '')
    val_str = val_series.astype(str).replace('nan', '')
    
    cond_df = cond_str.str.split(",", expand=True).replace("", np.nan)
    val_df = val_str.str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        out = np.nanmax(np.where(mask, val_arr, np.nan), axis=1)
    return pd.Series(out, index=cond_series.index)

def masked_min_from_csv_series(cond_series, val_series, mask_val):
    """
    Calculate masked minimum from CSV series.
    
    Args:
        cond_series: Condition series (comma-separated)
        val_series: Value series (comma-separated)
        mask_val: Value to mask on
        
    Returns:
        pd.Series: Masked min values
    """
    cond_str = cond_series.astype(str).replace('nan', '')
    val_str = val_series.astype(str).replace('nan', '')
    
    cond_df = cond_str.str.split(",", expand=True).replace("", np.nan)
    val_df = val_str.str.split(",", expand=True).replace("", np.nan)
    cond_arr = cond_df.to_numpy(dtype=float)
    val_arr = val_df.to_numpy(dtype=float)
    mask = (cond_arr == mask_val)
    with np.errstate(invalid="ignore"):
        out = np.nanmin(np.where(mask, val_arr, np.nan), axis=1)
    return pd.Series(out, index=cond_series.index)