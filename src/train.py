# train.py
# Model training script with cross-validation

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
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.features import FeatureEngineer
from src.utils import setup_logging

def train_model(X_train, y_train, X_val, y_val, params, test_type='A'):
    """
    Train single LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Model hyperparameters
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
            lgb.early_stopping(stopping_rounds=100, verbose=True)
        ]
    )
    
    return model

def cross_validate(X, y, params, n_splits=5, test_type='A'):
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        X: Features
        y: Labels
        params: Model hyperparameters
        n_splits: Number of folds
        test_type: 'A' or 'B'
        
    Returns:
        tuple: (best_model, cv_scores, oof_predictions)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    oof_preds = np.zeros(len(X))
    models = []
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {n_splits}-fold CV for type {test_type}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Fold {fold}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
        logger.info(f"Train pos rate: {y_train.mean():.4f}, Val pos rate: {y_val.mean():.4f}")
        
        model = train_model(X_train, y_train, X_val, y_val, params, test_type)
        models.append(model)
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_val, val_pred)
        cv_scores.append(fold_auc)
        logger.info(f"Fold {fold} AUC: {fold_auc:.6f}")
    
    oof_auc = roc_auc_score(y, oof_preds)
    logger.info(f"Overall OOF AUC: {oof_auc:.6f}")
    logger.info(f"CV AUC: {np.mean(cv_scores):.6f} +/- {np.std(cv_scores):.6f}")
    
    best_fold_idx = np.argmax(cv_scores)
    best_model = models[best_fold_idx]
    logger.info(f"Best model from fold {best_fold_idx + 1} (AUC: {cv_scores[best_fold_idx]:.6f})")
    
    return best_model, cv_scores, oof_preds

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

def main():
    """
    Main training execution pipeline.
    
    Workflow:
        1. Load training data
        2. Preprocess by test type
        3. Generate features
        4. Train with cross-validation
        5. Save models
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
        logger.info(f"Feature names: {len(X_a.columns)}")
        
        model_a, cv_scores_a, oof_preds_a = cross_validate(
            X_a, y_a, 
            config.lgbm_params_a,
            n_splits=config.training['n_splits'],
            test_type='A'
        )
        
        model_path_a = config.paths['model']
        joblib.dump(model_a, model_path_a)
        logger.info(f"Model A saved to {model_path_a}")
        
        logger.info("="*60)
        logger.info("STEP 5: Training Type B model")
        logger.info("="*60)
        
        X_b = prepare_features(featured_b, 'B')
        y_b = featured_b['Label'].astype(int)
        logger.info(f"Features B: {X_b.shape}")
        logger.info(f"Feature names: {len(X_b.columns)}")
        
        model_b, cv_scores_b, oof_preds_b = cross_validate(
            X_b, y_b,
            config.lgbm_params_b,
            n_splits=config.training['n_splits'],
            test_type='B'
        )
        
        model_path_b = config.paths['model_b']
        joblib.dump(model_b, model_path_b)
        logger.info(f"Model B saved to {model_path_b}")
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Type A - CV AUC: {np.mean(cv_scores_a):.6f} +/- {np.std(cv_scores_a):.6f}")
        logger.info(f"Type B - CV AUC: {np.mean(cv_scores_b):.6f} +/- {np.std(cv_scores_b):.6f}")
        
        total_time = time.time() - start_time
        logger.info(f"Total training time: {total_time:.1f}s ({total_time/60:.1f}min)")
        logger.info("="*60)
        
        feature_importance_a = pd.DataFrame({
            'feature': X_a.columns,
            'importance': model_a.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        feature_importance_b = pd.DataFrame({
            'feature': X_b.columns,
            'importance': model_b.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 features for Type A:")
        for idx, row in feature_importance_a.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.1f}")
        
        logger.info("Top 10 features for Type B:")
        for idx, row in feature_importance_b.head(10).iterrows():
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