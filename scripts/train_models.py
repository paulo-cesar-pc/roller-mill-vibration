#!/usr/bin/env python3
"""
Main training script for the roller mill vibration prediction project.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import get_config
from src.data.data_loader import DataLoader, DataSplitter
from src.data.validators import DataValidationPipeline, create_default_validators
from src.features.feature_engineer import create_default_pipeline
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import evaluate_model_comprehensive


def setup_logging():
    """Set up logging configuration."""
    config = get_config()
    
    # Create logs directory
    log_dir = Path(config.paths.logs)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    logger = setup_logging()
    logger.info("Starting roller mill vibration prediction training pipeline")
    
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Loaded configuration: {config.project.name} v{config.project.version}")
        
        # 1. Data Loading and Validation
        logger.info("=" * 60)
        logger.info("STEP 1: DATA LOADING AND VALIDATION")
        logger.info("=" * 60)
        
        data_loader = DataLoader()
        
        # Check if processed data exists
        processed_data_path = Path(config.data.processed_data_path) / "processed_data.csv"
        
        if processed_data_path.exists():
            logger.info(f"Loading existing processed data from {processed_data_path}")
            df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
            
            # Quick quality check
            quality_report = data_loader.validate_data_quality(df)
            logger.info(f"Data quality score: {quality_report.quality_score:.1f}/100")
        else:
            logger.info("Loading and processing raw data")
            df, quality_report = data_loader.load_and_process(save_processed=True, testing_mode=True)
            
            if quality_report.quality_score < 50:
                logger.error(f"Data quality too low: {quality_report.quality_score:.1f}/100")
                logger.error(f"Issues: {quality_report.issues}")
                return 1
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Data validation
        validators = create_default_validators(config)
        validation_pipeline = DataValidationPipeline(validators)
        validation_results = validation_pipeline.validate(df)
        validation_summary = validation_pipeline.get_summary(validation_results)
        
        logger.info(f"Data validation: {validation_summary['passed']}/{validation_summary['total_validators']} passed")
        
        if not validation_summary['all_passed']:
            logger.warning(f"Validation warnings: {validation_summary['warnings']}")
            if validation_summary['errors']:
                logger.error(f"Validation errors: {validation_summary['errors']}")
            
            # Only stop if there are actual errors (not just warnings) and success rate is very low
            if len(validation_summary['errors']) > 0 and validation_summary['success_rate'] < 50:
                logger.error("Critical validation failures. Stopping training.")
                return 1
            else:
                logger.info("Proceeding with training despite validation warnings.")
        
        # 2. Feature Engineering
        logger.info("=" * 60)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        # Create feature engineering pipeline
        feature_pipeline = create_default_pipeline(config)
        
        # Separate target variable
        target_col = config.data.target_column
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            return 1
        
        # Apply feature engineering
        logger.info("Applying feature engineering transformations")
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
        
        # Fit and transform features
        X_engineered = feature_pipeline.fit_transform(X, y)
        
        logger.info(f"Feature engineering completed: {X.shape[1]} -> {X_engineered.shape[1]} features")
        
        # Remove rows with NaN values (from lag/rolling features)
        original_length = len(X_engineered)
        valid_mask = ~X_engineered.isnull().any(axis=1) & ~y.isnull()
        X_clean = X_engineered[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"Removed {original_length - len(X_clean)} rows with NaN values")
        logger.info(f"Final dataset shape: {X_clean.shape}")
        
        # 3. Data Splitting
        logger.info("=" * 60)
        logger.info("STEP 3: DATA SPLITTING")
        logger.info("=" * 60)
        
        splitter = DataSplitter()
        
        # Create a temporary DataFrame for splitting
        temp_df = X_clean.copy()
        temp_df[target_col] = y_clean
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = splitter.time_series_split(temp_df, target_col)
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training: {len(X_train)} samples ({len(X_train)/len(X_clean)*100:.1f}%)")
        logger.info(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X_clean)*100:.1f}%)")
        logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/len(X_clean)*100:.1f}%)")
        
        # 4. Model Training
        logger.info("=" * 60)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 60)
        
        trainer = ModelTrainer()
        
        # Train all models with hyperparameter optimization
        training_results = trainer.train_all_models(
            X_train, y_train, X_val, y_val,
            optimize_hyperparams=True,
            create_ensembles=True
        )
        
        logger.info(f"Training completed: {training_results['total_models_trained']} models trained")
        
        # Get best model
        best_model = training_results.get('best_model')
        if best_model:
            val_metrics = best_model.evaluate(X_val, y_val)
            logger.info(f"Best model: {best_model.name}")
            logger.info(f"  Validation R²: {val_metrics.get('r2_score', 0):.4f}")
            logger.info(f"  Validation RMSE: {val_metrics.get('rmse', 0):.4f}")
        
        # 5. Model Evaluation
        logger.info("=" * 60)
        logger.info("STEP 5: MODEL EVALUATION")
        logger.info("=" * 60)
        
        if best_model:
            # Comprehensive evaluation on test set
            evaluation_results = evaluate_model_comprehensive(
                best_model,
                X_test,
                y_test,
                model_name=best_model.name,
                time_index=X_test.index,
                save_plots=True,
                save_report=True
            )
            
            test_metrics = evaluation_results['regression_metrics']
            logger.info(f"Test set performance:")
            logger.info(f"  R²: {test_metrics.get('r2_score', 0):.4f}")
            logger.info(f"  RMSE: {test_metrics.get('rmse', 0):.4f}")
            logger.info(f"  MAE: {test_metrics.get('mae', 0):.4f}")
            if 'mape' in test_metrics:
                logger.info(f"  MAPE: {test_metrics.get('mape', 0):.2f}%")
        
        # 6. Save Results
        logger.info("=" * 60)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("=" * 60)
        
        # Save training results
        results_path = Path(config.paths.experiments) / "training_results.json"
        trainer.save_training_results(training_results, results_path)
        
        # Save best model separately
        if best_model:
            best_model_path = Path(config.paths.models) / f"best_model_{best_model.name}.pkl"
            best_model.save_model(best_model_path)
            logger.info(f"Best model saved to: {best_model_path}")
        
        # Save feature pipeline
        import joblib
        pipeline_path = Path(config.paths.models) / "feature_pipeline.pkl"
        joblib.dump(feature_pipeline, pipeline_path)
        logger.info(f"Feature pipeline saved to: {pipeline_path}")
        
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
        # Print summary
        logger.info("SUMMARY:")
        logger.info(f"  Dataset: {len(X_clean):,} samples, {X_clean.shape[1]} features")
        logger.info(f"  Models trained: {training_results['total_models_trained']}")
        logger.info(f"  Best model: {best_model.name if best_model else 'None'}")
        if best_model and test_metrics:
            logger.info(f"  Test R²: {test_metrics.get('r2_score', 0):.4f}")
        logger.info(f"  Results saved to: {config.paths.experiments}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)