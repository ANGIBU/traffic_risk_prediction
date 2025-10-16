# src/logger.py
# Result logging system for experiment tracking

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

class ResultLogger:
    """
    Experiment result logger for tracking inference runs.
    
    Logs key parameters, model settings, and prediction statistics
    to a persistent text file for research and analysis purposes.
    """
    
    def __init__(self, log_path: Path):
        """
        Initialize result logger.
        
        Args:
            log_path: Path to log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment counter
        self.experiment_number = self._get_next_experiment_number()
    
    def _get_next_experiment_number(self) -> int:
        """
        Get next experiment number by counting previous entries.
        
        Returns:
            int: Next experiment number
        """
        if not self.log_path.exists():
            return 1
        
        with open(self.log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            count = content.count('Experiment #')
            return count + 1
    
    def log_experiment(self, 
                      config: Any,
                      results: Dict[str, Any],
                      execution_time: float,
                      notes: str = "") -> None:
        """
        Log experiment results to file.
        
        Args:
            config: Configuration object
            results: Dictionary containing prediction results and statistics
            execution_time: Total execution time in seconds
            notes: Optional notes about the experiment
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_content = self._format_log_entry(
            experiment_number=self.experiment_number,
            timestamp=timestamp,
            config=config,
            results=results,
            execution_time=execution_time,
            notes=notes
        )
        
        # Append to log file
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_content)
        
        print(f"Results logged to {self.log_path}")
    
    def _format_log_entry(self,
                         experiment_number: int,
                         timestamp: str,
                         config: Any,
                         results: Dict[str, Any],
                         execution_time: float,
                         notes: str) -> str:
        """
        Format log entry with structured information.
        
        Args:
            experiment_number: Experiment number
            timestamp: Timestamp string
            config: Configuration object
            results: Results dictionary
            execution_time: Execution time in seconds
            notes: Additional notes
            
        Returns:
            str: Formatted log entry
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"[{timestamp}] Experiment #{experiment_number}")
        lines.append("=" * 70)
        lines.append("")
        
        if notes:
            lines.append(f"Notes: {notes}")
            lines.append("")
        
        # Key Parameters
        lines.append("--- Key Parameters ---")
        params = self._extract_parameters(config)
        for key, value in sorted(params.items()):
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Results
        lines.append("--- Results ---")
        lines.append("")
        
        # Type A predictions
        if 'predictions_a' in results and len(results['predictions_a']) > 0:
            pred_a = results['predictions_a']
            lines.append("  [Type A Predictions]")
            lines.append(f"    Samples: {len(pred_a)}")
            lines.append(f"    Mean: {np.mean(pred_a):.6f}")
            lines.append(f"    Std: {np.std(pred_a):.6f}")
            lines.append(f"    Min: {np.min(pred_a):.6f}")
            lines.append(f"    Max: {np.max(pred_a):.6f}")
            lines.append("")
        
        # Type B predictions
        if 'predictions_b' in results and len(results['predictions_b']) > 0:
            pred_b = results['predictions_b']
            lines.append("  [Type B Predictions]")
            lines.append(f"    Samples: {len(pred_b)}")
            lines.append(f"    Mean: {np.mean(pred_b):.6f}")
            lines.append(f"    Std: {np.std(pred_b):.6f}")
            lines.append(f"    Min: {np.min(pred_b):.6f}")
            lines.append(f"    Max: {np.max(pred_b):.6f}")
            lines.append("")
        
        # Overall predictions
        if 'predictions_all' in results:
            pred_all = results['predictions_all']
            lines.append("  [Overall Predictions]")
            lines.append(f"    Total Samples: {len(pred_all)}")
            lines.append(f"    Mean: {np.mean(pred_all):.6f}")
            lines.append(f"    Std: {np.std(pred_all):.6f}")
            lines.append(f"    Min: {np.min(pred_all):.6f}")
            lines.append(f"    Max: {np.max(pred_all):.6f}")
            lines.append("")
        
        # Model information
        lines.append("  [Models]")
        if 'model_a_loaded' in results:
            lines.append(f"    Type A Model: {'Loaded' if results['model_a_loaded'] else 'Not Found'}")
        if 'model_b_loaded' in results:
            lines.append(f"    Type B Model: {'Loaded' if results['model_b_loaded'] else 'Not Found'}")
        if 'feature_count_a' in results:
            lines.append(f"    Type A Features: {results['feature_count_a']}")
        if 'feature_count_b' in results:
            lines.append(f"    Type B Features: {results['feature_count_b']}")
        lines.append("")
        
        # Execution metrics
        lines.append("  [Execution Metrics]")
        lines.append(f"    Total Time: {execution_time:.1f}s")
        if 'load_time' in results:
            lines.append(f"    Data Load Time: {results['load_time']:.1f}s")
        if 'preprocess_time' in results:
            lines.append(f"    Preprocess Time: {results['preprocess_time']:.1f}s")
        if 'feature_time' in results:
            lines.append(f"    Feature Engineering Time: {results['feature_time']:.1f}s")
        if 'predict_time' in results:
            lines.append(f"    Prediction Time: {results['predict_time']:.1f}s")
        lines.append("")
        
        lines.append("=" * 70)
        lines.append("")
        
        return "\n".join(lines)
    
    def _extract_parameters(self, config: Any) -> Dict[str, Any]:
        """
        Extract key parameters from configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            dict: Parameter dictionary
        """
        params = {}
        
        # Inference parameters
        if hasattr(config, 'inference'):
            params['batch_size'] = config.inference.get('batch_size', 'N/A')
            params['num_threads'] = config.inference.get('num_threads', 'N/A')
            params['use_progress'] = config.inference.get('use_progress', 'N/A')
        
        # Preprocessing parameters
        if hasattr(config, 'preprocessing'):
            params['handle_missing'] = config.preprocessing.get('handle_missing', 'N/A')
            params['eps'] = config.preprocessing.get('eps', 'N/A')
        
        # Model paths
        if hasattr(config, 'paths'):
            if 'model' in config.paths:
                params['model_a_path'] = config.paths['model'].name
            if 'model_b' in config.paths:
                params['model_b_path'] = config.paths['model_b'].name
        
        return params