# script.py
# Main inference script for traffic accident risk prediction

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.config import Config
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.features import FeatureEngineer
from src.predictor import Predictor
from src.utils import setup_logging, validate_predictions

def main():
    """
    Main inference execution pipeline.
    
    Workflow:
        1. Initialize configuration
        2. Load test data
        3. Preprocess data by test type
        4. Generate features
        5. Generate predictions
        6. Save results
    """
    # Setup logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("Starting inference pipeline")
    logger.info("="*60)
    
    try:
        # Initialize configuration
        config = Config()
        config.validate()
        logger.info(f"Configuration: {config.get_summary()}")
        
        # Initialize components
        data_loader = DataLoader(config)
        preprocessor = Preprocessor(config)
        feature_engineer = FeatureEngineer(config)
        predictor = Predictor(config)
        
        # Load test data
        logger.info("="*60)
        logger.info("STEP 1: Loading test data")
        logger.info("="*60)
        test_meta, test_a, test_b = data_loader.load_test_data()
        logger.info(f"Loaded test metadata: {len(test_meta)} samples")
        
        # Merge metadata with detailed test data
        merged_a, merged_b = data_loader.merge_test_data(test_meta, test_a, test_b)
        logger.info(f"Type A samples: {len(merged_a)}")
        logger.info(f"Type B samples: {len(merged_b)}")
        
        # Preprocess data
        logger.info("="*60)
        logger.info("STEP 2: Preprocessing data")
        logger.info("="*60)
        processed_a = preprocessor.preprocess_a(merged_a)
        processed_b = preprocessor.preprocess_b(merged_b)
        logger.info(f"Preprocessed A: {processed_a.shape}")
        logger.info(f"Preprocessed B: {processed_b.shape}")
        
        # Generate features
        logger.info("="*60)
        logger.info("STEP 3: Feature engineering")
        logger.info("="*60)
        featured_a = feature_engineer.add_features_a(processed_a)
        featured_b = feature_engineer.add_features_b(processed_b)
        logger.info(f"Features A: {featured_a.shape}")
        logger.info(f"Features B: {featured_b.shape}")
        
        # Generate predictions
        logger.info("="*60)
        logger.info("STEP 4: Generating predictions")
        logger.info("="*60)
        predictions = predictor.predict(featured_a, featured_b)
        
        # Combine predictions
        logger.info("="*60)
        logger.info("STEP 5: Combining and saving results")
        logger.info("="*60)
        
        # Create submission dataframe
        submission_parts = []
        
        if len(predictions['A']) > 0:
            sub_a = pd.DataFrame({
                'Test_id': featured_a['Test_id'].values,
                'Label': predictions['A']
            })
            submission_parts.append(sub_a)
            logger.info(f"Type A predictions: {len(sub_a)}")
        
        if len(predictions['B']) > 0:
            sub_b = pd.DataFrame({
                'Test_id': featured_b['Test_id'].values,
                'Label': predictions['B']
            })
            submission_parts.append(sub_b)
            logger.info(f"Type B predictions: {len(sub_b)}")
        
        # Combine all predictions
        submission = pd.concat(submission_parts, axis=0, ignore_index=True)
        logger.info(f"Total predictions: {len(submission)}")
        
        # Merge with test metadata to ensure correct order
        final_submission = test_meta[['Test_id']].merge(
            submission, on='Test_id', how='left'
        )
        
        # Fill any missing predictions with 0.0
        final_submission['Label'] = final_submission['Label'].fillna(0.0)
        
        # Validate predictions
        validate_predictions(final_submission['Label'].values, len(test_meta))
        
        # Ensure output directory exists
        config.ensure_output_dir()
        
        # Save results
        output_path = config.paths['output']
        logger.info(f"Saving results to {output_path}")
        final_submission.to_csv(output_path, index=False)
        
        # Print summary statistics
        logger.info("="*60)
        logger.info("INFERENCE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total samples: {len(final_submission)}")
        logger.info(f"Prediction mean: {final_submission['Label'].mean():.6f}")
        logger.info(f"Prediction std: {final_submission['Label'].std():.6f}")
        logger.info(f"Prediction min: {final_submission['Label'].min():.6f}")
        logger.info(f"Prediction max: {final_submission['Label'].max():.6f}")
        logger.info(f"Output saved: {output_path}")
        logger.info("="*60)
        
        return 0
        
    except Exception as e:
        logger.error("="*60)
        logger.error("PIPELINE FAILED")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
