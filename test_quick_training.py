#!/usr/bin/env python3
"""
Quick test script to isolate training bottlenecks.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from config.settings import get_config
from src.data.data_loader import DataLoader
from src.models.integrated_noisy_trainer import IntegratedNoisyTrainer, create_integrated_config

def setup_logging():
    """Set up simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def main():
    """Quick test of training components."""
    logger = setup_logging()
    logger.info("Starting quick training test")
    
    try:
        # 1. Load minimal data
        logger.info("Loading data...")
        config = get_config()
        data_loader = DataLoader()
        df, quality_report = data_loader.load_and_process(testing_mode=True)
        
        # Limit to 1000 rows for speed
        df = df.head(1000)
        target_col = config.data.target_column
        logger.info(f"Dataset: {df.shape}, Target: {target_col}")
        
        # 2. Create minimal config
        logger.info("Creating minimal config...")
        noise_config = create_integrated_config(
            target_frequency='5min',
            focus_on_robustness=False,
            include_all_formulations=False  # Disable alternative formulations
        )
        # Override to minimal settings
        noise_config.create_mill_features = False  # Disable feature engineering
        noise_config.max_features = 50
        noise_config.n_validation_folds = 2
        noise_config.create_evaluation_plots = False
        
        # 3. Test integrated trainer
        logger.info("Testing integrated trainer...")
        trainer = IntegratedNoisyTrainer(noise_config)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col != target_col]
        logger.info(f"Features: {len(feature_cols)}")
        
        # 4. Try just preprocessing
        logger.info("Testing preprocessing...")
        trainer._setup_components()
        X_processed, y_processed = trainer.preprocess_data(df, target_col, feature_cols)
        logger.info(f"Preprocessed: {X_processed.shape}")
        
        logger.info("✅ Quick test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Quick test failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)