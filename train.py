# train.py
# Model training script with cross-validation and calibration
# Current target: 0.17 or below

import sys
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.feature_selection import SelectFromModel
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb

sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.features import FeatureEngineer
from src.utils import setup_logging

def calculate_ece(y_true, y_pred, n_bins=10):
    """
    Calculate Expected Calibration Error.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        float: ECE score
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece

def calculate_combined_score(y_true, y_pred):
    """
    Calculate combined score: 0.5*(1-AUC) + 0.25*Brier + 0.25*ECE.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        dict: Metric scores
    """
    auc = roc_auc_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    ece = calculate_ece(y_true, y_pred)
    combined = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece
    
    return {
        'auc': auc,
        'brier': brier,
        'ece': ece,
        'combined': combined
    }

def remove_correlated_features(X, threshold=0.95):
    """
    Remove highly correlated features.
    
    Args:
        X: Feature matrix
        threshold: Correlation threshold
        
    Returns:
        list: Features to keep
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Removing correlated features (threshold={threshold})")
    
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for column in upper.columns:
        if any(upper[column] > threshold):
            to_drop.append(column)
    
    features_to_keep = [f for f in X.columns if f not in to_drop]
    
    logger.info(f"Removed {len(to_drop)} correlated features")
    logger.info(f"Keeping {len(features_to_keep)} features")
    
    return features_to_keep

def train_model(X_train, y_train, X_val, y_val, params, early_stopping_rounds=50, test_type='A'):
    """
    Train single LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Model hyperparameters
        early_stopping_rounds: Early stopping rounds
        test_type: 'A' or 'B'
        
    Returns:
        Trained model
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)
        ]
    )
    
    return model

def select_features(X, y, params, threshold=0.90):
    """
    Select important features using feature importance.
    
    Args:
        X: Feature matrix
        y: Labels
        params: Model parameters
        threshold: Cumulative importance threshold
        
    Returns:
        list: Selected feature names
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting feature selection (threshold={threshold})")
    
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    importance = model.feature_importance(importance_type='gain')
    feature_names = X.columns.tolist()
    
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    feat_imp['cumulative'] = feat_imp['importance'].cumsum() / feat_imp['importance'].sum()
    
    selected = feat_imp[feat_imp['cumulative'] <= threshold]['feature'].tolist()
    
    min_features = max(30, int(len(feature_names) * 0.4))
    if len(selected) < min_features:
        selected = feat_imp.head(min_features)['feature'].tolist()
    
    logger.info(f"Selected {len(selected)} features from {len(feature_names)}")
    logger.info(f"Cumulative importance: {feat_imp.iloc[len(selected)-1]['cumulative']:.3f}")
    
    return selected

def cross_validate(X, y, params, config, test_type='A'):
    """
    Perform stratified k-fold cross-validation with ensemble capability.
    
    Args:
        X: Features
        y: Labels
        params: Model hyperparameters
        config: Configuration object
        test_type: 'A' or 'B'
        
    Returns:
        tuple: (ensemble_models, cv_scores, oof_predictions, selected_features, calibrator)
    """
    n_splits = config.training['n_splits']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {n_splits}-fold CV for type {test_type}")
    
    # Remove correlated features first
    if config.training['remove_correlated_features']:
        corr_threshold = config.feature_engineering['correlation_threshold']
        keep_features = remove_correlated_features(X, threshold=corr_threshold)
        X = X[keep_features]
        logger.info(f"After correlation removal: {len(X.columns)} features")
    
    # Feature selection with type-specific threshold
    if config.training['use_feature_selection']:
        if test_type == 'B':
            threshold = config.training.get('feature_selection_threshold_b', 0.85)
        else:
            threshold = config.training.get('feature_selection_threshold', 0.90)
        
        selected_features = select_features(X, y, params, threshold=threshold)
        X = X[selected_features]
        logger.info(f"Using {len(selected_features)} selected features")
    else:
        selected_features = X.columns.tolist()
    
    # Early stopping rounds based on test type
    if test_type == 'B':
        early_stopping_rounds = config.training.get('early_stopping_rounds_b', 100)
    else:
        early_stopping_rounds = config.training.get('early_stopping_rounds', 50)
    
    logger.info(f"Using early_stopping_rounds={early_stopping_rounds} for type {test_type}")
    
    cv_metrics = []
    oof_preds = np.zeros(len(X))
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Fold {fold}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"Train pos rate: {y_train.mean():.4f}, Val pos rate: {y_val.mean():.4f}")
        
        model = train_model(X_train, y_train, X_val, y_val, params, early_stopping_rounds, test_type)
        models.append(model)
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        fold_metrics = calculate_combined_score(y_val, val_pred)
        cv_metrics.append(fold_metrics)
        
        logger.info(f"Fold {fold} - AUC: {fold_metrics['auc']:.6f}, "
                   f"Brier: {fold_metrics['brier']:.6f}, "
                   f"ECE: {fold_metrics['ece']:.6f}, "
                   f"Combined: {fold_metrics['combined']:.6f}")
    
    oof_metrics = calculate_combined_score(y, oof_preds)
    logger.info(f"Overall OOF - AUC: {oof_metrics['auc']:.6f}, "
               f"Brier: {oof_metrics['brier']:.6f}, "
               f"ECE: {oof_metrics['ece']:.6f}, "
               f"Combined: {oof_metrics['combined']:.6f}")
    
    metrics_df = pd.DataFrame(cv_metrics)
    for metric in ['auc', 'brier', 'ece', 'combined']:
        mean = metrics_df[metric].mean()
        std = metrics_df[metric].std()
        logger.info(f"CV {metric.upper()}: {mean:.6f} +/- {std:.6f}")
    
    best_fold_idx = np.argmin([m['combined'] for m in cv_metrics])
    best_model = models[best_fold_idx]
    logger.info(f"Best model from fold {best_fold_idx + 1} "
               f"(Combined: {cv_metrics[best_fold_idx]['combined']:.6f})")
    
    # Use best model OOF for calibration
    oof_for_calibration = oof_preds
    
    # Calibrator (optional)
    calibrator = None
    if config.training['use_calibration']:
        logger.info("Training probability calibrator...")
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(oof_for_calibration, y)
        
        calibrated_preds = calibrator.transform(oof_for_calibration)
        calibrated_metrics = calculate_combined_score(y, calibrated_preds)
        
        logger.info(f"After calibration - AUC: {calibrated_metrics['auc']:.6f}, "
                   f"Brier: {calibrated_metrics['brier']:.6f}, "
                   f"ECE: {calibrated_metrics['ece']:.6f}, "
                   f"Combined: {calibrated_metrics['combined']:.6f}")
    else:
        logger.info("Calibration disabled")
    
    return best_model, cv_metrics, oof_preds, selected_features, calibrator

def prepare_features(df, test_type='A'):
    """
    Prepare feature matrix by removing non-feature columns.
    
    Args:
        df: DataFrame with features
        test_type: 'A' or 'B'
        
    Returns:
        pd.DataFrame: Feature matrix
    """
    drop_cols = ['Test_id', 'Test', 'PrimaryKey', 'Age', 'TestDate', 'Label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(0.0)
    
    return X

def save_feature_names(feature_names, path):
    """
    Save feature names to file.
    
    Args:
        feature_names: List of feature names
        path: File path
    """
    with open(path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")

def main():
    """
    Main training execution pipeline.
    
    Workflow:
        1. Load training data
        2. Preprocess by test type
        3. Generate features
        4. Train with cross-validation
        5. Save models and feature names
    """
    start_time = time.time()
    
    logger = setup_logging()
    logger.info("="*60)
    logger.info("Starting training pipeline")
    logger.info("="*60)
    
    try:
        config = Config(mode='training')
        config.validate()
        config.ensure_model_dir()
        
        data_loader = DataLoader(config)
        preprocessor = Preprocessor(config)
        feature_engineer = FeatureEngineer(config)
        
        logger.info("="*60)
        logger.info("STEP 1: Loading training data")
        logger.info("="*60)
        
        train_meta = pd.read_csv(config.paths['train'])
        logger.info(f"Loaded train metadata: {len(train_meta)} samples")
        logger.info(f"Positive rate: {train_meta['Label'].mean():.4f}")
        
        train_a_detail = pd.read_csv(config.paths['train_a'])
        train_b_detail = pd.read_csv(config.paths['train_b'])
        logger.info(f"Loaded A details: {len(train_a_detail)}")
        logger.info(f"Loaded B details: {len(train_b_detail)}")
        
        meta_a = train_meta[train_meta['Test'] == 'A'].copy()
        meta_b = train_meta[train_meta['Test'] == 'B'].copy()
        
        merged_a = meta_a.merge(train_a_detail, on='Test_id', how='left')
        merged_b = meta_b.merge(train_b_detail, on='Test_id', how='left')
        
        logger.info(f"Merged A: {merged_a.shape}")
        logger.info(f"Merged B: {merged_b.shape}")
        
        logger.info("="*60)
        logger.info("STEP 2: Preprocessing")
        logger.info("="*60)
        
        processed_a = preprocessor.preprocess_a(merged_a)
        processed_b = preprocessor.preprocess_b(merged_b)
        logger.info(f"Processed A: {processed_a.shape}")
        logger.info(f"Processed B: {processed_b.shape}")
        
        logger.info("="*60)
        logger.info("STEP 3: Feature engineering")
        logger.info("="*60)
        
        featured_a = feature_engineer.add_features_a(processed_a)
        featured_b = feature_engineer.add_features_b(processed_b)
        logger.info(f"Featured A: {featured_a.shape}")
        logger.info(f"Featured B: {featured_b.shape}")
        
        logger.info("="*60)
        logger.info("STEP 4: Training Type A model")
        logger.info("="*60)
        
        X_a = prepare_features(featured_a, 'A')
        y_a = featured_a['Label'].astype(int)
        logger.info(f"Features A: {X_a.shape}")
        logger.info(f"Initial feature count: {len(X_a.columns)}")
        
        ensemble_models_a, cv_metrics_a, oof_preds_a, selected_features_a, calibrator_a = cross_validate(
            X_a, y_a, 
            config.lgbm_params_a,
            config,
            test_type='A'
        )
        
        # Save model
        model_path_a = config.paths['model']
        joblib.dump(ensemble_models_a, model_path_a)
        logger.info(f"Model A saved to {model_path_a}")
        
        if calibrator_a is not None:
            calibrator_path_a = config.paths['model_dir'] / 'calibrator_A.pkl'
            joblib.dump(calibrator_a, calibrator_path_a)
            logger.info(f"Calibrator A saved to {calibrator_path_a}")
        
        feature_names_path_a = config.paths['feature_names_a']
        save_feature_names(selected_features_a, feature_names_path_a)
        logger.info(f"Feature names A saved to {feature_names_path_a}")
        
        logger.info("="*60)
        logger.info("STEP 5: Training Type B model")
        logger.info("="*60)
        
        X_b = prepare_features(featured_b, 'B')
        y_b = featured_b['Label'].astype(int)
        logger.info(f"Features B: {X_b.shape}")
        logger.info(f"Initial feature count: {len(X_b.columns)}")
        
        ensemble_models_b, cv_metrics_b, oof_preds_b, selected_features_b, calibrator_b = cross_validate(
            X_b, y_b,
            config.lgbm_params_b,
            config,
            test_type='B'
        )
        
        # Save model
        model_path_b = config.paths['model_b']
        joblib.dump(ensemble_models_b, model_path_b)
        logger.info(f"Model B saved to {model_path_b}")
        
        if calibrator_b is not None:
            calibrator_path_b = config.paths['model_dir'] / 'calibrator_B.pkl'
            joblib.dump(calibrator_b, calibrator_path_b)
            logger.info(f"Calibrator B saved to {calibrator_path_b}")
        
        feature_names_path_b = config.paths['feature_names_b']
        save_feature_names(selected_features_b, feature_names_path_b)
        logger.info(f"Feature names B saved to {feature_names_path_b}")
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        
        metrics_df_a = pd.DataFrame(cv_metrics_a)
        metrics_df_b = pd.DataFrame(cv_metrics_b)
        
        logger.info(f"Type A - Combined Score: {metrics_df_a['combined'].mean():.6f} +/- {metrics_df_a['combined'].std():.6f}")
        logger.info(f"Type A - AUC: {metrics_df_a['auc'].mean():.6f} +/- {metrics_df_a['auc'].std():.6f}")
        logger.info(f"Type B - Combined Score: {metrics_df_b['combined'].mean():.6f} +/- {metrics_df_b['combined'].std():.6f}")
        logger.info(f"Type B - AUC: {metrics_df_b['auc'].mean():.6f} +/- {metrics_df_b['auc'].std():.6f}")
        
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info("="*60)
        
        # Feature importance logging
        feature_importance_a = pd.DataFrame({
            'feature': selected_features_a,
            'importance': ensemble_models_a.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        feature_importance_b = pd.DataFrame({
            'feature': selected_features_b,
            'importance': ensemble_models_b.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 15 features for Type A:")
        for idx, row in feature_importance_a.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.1f}")
        
        logger.info("Top 15 features for Type B:")
        for idx, row in feature_importance_b.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.1f}")
        
        config.ensure_output_dir()
        feature_importance_a.to_csv(config.paths['output_dir'] / 'feature_importance_A.csv', index=False)
        feature_importance_b.to_csv(config.paths['output_dir'] / 'feature_importance_B.csv', index=False)
        logger.info("Feature importance saved")
        
        return 0
        
    except Exception as e:
        logger.error("="*60)
        logger.error("TRAINING FAILED")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())