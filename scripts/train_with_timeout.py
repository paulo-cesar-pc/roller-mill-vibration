#!/usr/bin/env python3
"""
Training script with timeout and progress monitoring for large datasets.
"""

import logging
import sys
import time
import signal
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from contextlib import contextmanager

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import get_config
from src.models.smart_trainer import SmartModelTrainer
from src.data.data_loader import DataLoader
from src.evaluation.time_series_evaluator import TimeSeriesEvaluator


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(duration):
    """Context manager for timeout functionality."""
    def timeout_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {duration} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)


def setup_logging():
    """Set up logging configuration with more frequent progress updates."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_data_with_timeout(logger, timeout_seconds=300, test_sample_size=1000):
    """Load data with timeout protection and optional sampling for testing."""
    logger.info("Loading data with timeout protection...")
    
    data_paths = [
        Path("data/processed/processed_data.csv"),
        Path("data/raw/roller_mill_data.csv"), 
        Path("full_data/roller_mill_data.csv")
    ]
    
    df = None
    for path in data_paths:
        if path.exists():
            logger.info(f"Attempting to load data from {path}")
            try:
                with timeout(timeout_seconds):
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                    logger.info(f"Successfully loaded data with shape: {df.shape}")
                    break
            except TimeoutException:
                logger.error(f"Loading {path} timed out after {timeout_seconds} seconds")
                continue
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue
    
    if df is None:
        logger.info("Attempting to load data using DataLoader...")
        try:
            with timeout(timeout_seconds):
                data_loader = DataLoader()
                df = data_loader.load_and_preprocess()
                logger.info(f"DataLoader successful: {df.shape}")
        except TimeoutException:
            logger.error(f"DataLoader timed out after {timeout_seconds} seconds")
            raise
        except Exception as e:
            logger.error(f"DataLoader failed: {e}")
            raise FileNotFoundError("No data file found or loading failed")
    
    # Limit dataset for testing
    if test_sample_size and len(df) > test_sample_size:
        logger.info(f"🧪 TEST MODE: Limiting dataset to {test_sample_size:,} rows for faster testing")
        logger.info(f"   Original size: {len(df):,} rows")
        
        # Sample from the middle to get more representative data
        start_idx = len(df) // 4  # Start at 25% through the data
        end_idx = start_idx + test_sample_size
        
        if end_idx > len(df):
            end_idx = len(df)
            start_idx = max(0, end_idx - test_sample_size)
        
        df = df.iloc[start_idx:end_idx].copy()
        logger.info(f"   Test sample: {len(df):,} rows from index {start_idx} to {end_idx}")
        logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def quick_data_check(df, target_col, logger):
    """Perform quick data quality checks."""
    logger.info("Performing quick data quality check...")
    
    # Basic checks
    n_samples = len(df)
    n_features = len(df.columns)
    logger.info(f"Dataset: {n_samples:,} samples × {n_features} features")
    
    # Target check
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found!")
        return False
    
    target_stats = df[target_col].describe()
    logger.info(f"Target statistics: mean={target_stats['mean']:.3f}, std={target_stats['std']:.3f}")
    
    # Check for target-like columns
    target_like_columns = []
    if 'CM2_PV_VRM01_VIBRATION' in target_col.upper():
        for col in df.columns:
            if 'VIBRATION' in col.upper() and col != target_col:
                target_like_columns.append(col)
    
    if target_like_columns:
        logger.warning(f"Found {len(target_like_columns)} potential target-like columns")
        logger.warning(f"Will exclude: {target_like_columns[:3]}{'...' if len(target_like_columns) > 3 else ''}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"Dataset memory usage: {memory_mb:.1f} MB")
    
    return True


def train_with_progress_monitoring(df, target_col, logger):
    """Train with progress monitoring and timeout protection."""
    
    logger.info("=" * 60)
    logger.info("STARTING INTELLIGENT TRAINING WITH PROGRESS MONITORING")
    logger.info("=" * 60)
    
    # Initialize trainer
    logger.info("Initializing SmartModelTrainer...")
    trainer = SmartModelTrainer()
    
    # Training with timeout
    training_timeout = 1800  # 30 minutes
    logger.info(f"Training will timeout after {training_timeout} seconds")
    
    try:
        with timeout(training_timeout):
            logger.info("Starting fit process...")
            start_time = time.time()
            
            trainer.fit(df, target_col)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Training completed in {elapsed_time:.1f} seconds")
            
    except TimeoutException:
        logger.error(f"Training timed out after {training_timeout} seconds")
        logger.error("This usually indicates:")
        logger.error("1. Dataset is too large for current memory")
        logger.error("2. Feature engineering is taking too long") 
        logger.error("3. Model training is stuck")
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return None
    
    return trainer


def quick_evaluation(trainer, logger):
    """Quick evaluation without comprehensive plots."""
    logger.info("=" * 60)
    logger.info("QUICK EVALUATION")
    logger.info("=" * 60)
    
    try:
        # Get test results
        test_results = trainer.evaluate_on_test()
        
        logger.info("Quick Test Results:")
        logger.info(f"  Best Model: {test_results['model_name']}")
        logger.info(f"  R² Score: {test_results['test_r2']:.4f}")
        logger.info(f"  RMSE: {test_results['test_rmse']:.4f}")
        logger.info(f"  MAE: {test_results['test_mae']:.4f}")
        logger.info(f"  MAPE: {test_results['test_mape']:.2f}%")
        
        # Training summary
        summary = trainer.get_training_summary()
        if summary:
            logger.info(f"  Features Used: {summary['num_features']}")
            logger.info(f"  Training Samples: {summary['training_samples']:,}")
            logger.info(f"  Test Samples: {summary['test_samples']:,}")
        
        # Performance assessment
        if test_results['test_r2'] > 0.5:
            logger.info("🎉 EXCELLENT: Model achieved good performance!")
        elif test_results['test_r2'] > 0.2:
            logger.info("✅ GOOD: Model shows reasonable performance!")
        elif test_results['test_r2'] > 0:
            logger.info("⚠️  ACCEPTABLE: Model is working but could be improved")
        else:
            logger.info("❌ POOR: Model has serious issues")
        
        return test_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return None


def main():
    """Main training function with timeout protection."""
    logger = setup_logging()
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        # Load configuration
        config = get_config()
        target_col = config.data.target_column
        logger.info(f"Target column: {target_col}")
        
        # Step 1: Load data with timeout
        logger.info("=" * 60)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 60)
        
        df = load_data_with_timeout(logger, timeout_seconds=300, test_sample_size=1000)
        
        # Step 2: Quick data check
        logger.info("=" * 60) 
        logger.info("STEP 2: DATA QUALITY CHECK")
        logger.info("=" * 60)
        
        if not quick_data_check(df, target_col, logger):
            logger.error("Data quality check failed")
            return 1
        
        # Step 3: Training with monitoring
        logger.info("=" * 60)
        logger.info("STEP 3: INTELLIGENT TRAINING")  
        logger.info("=" * 60)
        
        trainer = train_with_progress_monitoring(df, target_col, logger)
        if trainer is None:
            logger.error("Training failed")
            return 1
        
        # Step 4: Quick evaluation
        logger.info("=" * 60)
        logger.info("STEP 4: EVALUATION")
        logger.info("=" * 60)
        
        results = quick_evaluation(trainer, logger)
        if results is None:
            logger.error("Evaluation failed")
            return 1
        
        # Step 5: Save model
        logger.info("=" * 60)
        logger.info("STEP 5: SAVING MODEL")
        logger.info("=" * 60)
        
        output_dir = Path("outputs/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_model(str(model_path))
        logger.info(f"Model saved to: {model_path}")
        
        # Final summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final R² Score: {results['test_r2']:.4f}")
        logger.info(f"Best Model: {results['model_name']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Handle Windows signal limitations
    if sys.platform != 'win32':
        exit_code = main()
    else:
        # On Windows, run without signal timeout
        logger = setup_logging()
        logger.info("Running on Windows - timeout protection disabled")
        exit_code = main()
    
    sys.exit(exit_code)