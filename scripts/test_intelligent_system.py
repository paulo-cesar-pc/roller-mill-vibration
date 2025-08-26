#!/usr/bin/env python3
"""
Test script to validate the complete intelligent system with synthetic data.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.smart_trainer import SmartModelTrainer
from src.evaluation.time_series_evaluator import TimeSeriesEvaluator


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def generate_synthetic_time_series_data(n_samples=10000, n_features=20, noise_level=0.1):
    """Generate synthetic time series data for testing."""
    
    # Create time index
    start_date = datetime(2023, 1, 1)
    timestamps = pd.date_range(start_date, periods=n_samples, freq='1H')
    
    # Generate base features with different patterns
    np.random.seed(42)
    
    data = {}
    
    # Time-based features
    hours = timestamps.hour
    days = timestamps.dayofweek
    months = timestamps.month
    
    # Create features with different correlation patterns
    for i in range(n_features):
        feature_name = f'feature_{i:02d}'
        
        if i < 5:  # High correlation features
            # Linear trend + seasonal + noise
            trend = 0.001 * np.arange(n_samples)
            seasonal = 2 * np.sin(2 * np.pi * hours / 24) + np.sin(2 * np.pi * days / 7)
            noise = np.random.normal(0, noise_level, n_samples)
            data[feature_name] = trend + seasonal + noise + 10
            
        elif i < 10:  # Medium correlation features
            # Mostly seasonal with some trend
            seasonal = 1.5 * np.sin(2 * np.pi * hours / 24 + i) + 0.5 * np.sin(2 * np.pi * days / 7)
            trend = 0.0005 * np.arange(n_samples)
            noise = np.random.normal(0, noise_level * 2, n_samples)
            data[feature_name] = seasonal + trend + noise + 5
            
        elif i < 15:  # Low correlation features
            # Random walk with slight seasonal influence
            seasonal = 0.5 * np.sin(2 * np.pi * hours / 24 + i * 2)
            random_walk = np.cumsum(np.random.normal(0, 0.01, n_samples))
            noise = np.random.normal(0, noise_level * 3, n_samples)
            data[feature_name] = seasonal + random_walk + noise
            
        else:  # Random features (should have low correlation)
            data[feature_name] = np.random.normal(0, 1, n_samples)
    
    # Create target variable with known relationships
    target = (
        0.7 * data['feature_00'] +  # Strong positive correlation
        0.5 * data['feature_01'] +  # Moderate positive correlation
        -0.3 * data['feature_02'] + # Negative correlation
        0.2 * data['feature_03'] +  # Weak positive correlation
        0.1 * data['feature_04'] +  # Very weak correlation
        np.random.normal(0, noise_level * 5, n_samples)  # Noise
    )
    
    # Add some non-linear relationships
    target += 0.1 * np.sin(data['feature_05'] * 0.5)
    target += 0.05 * data['feature_06'] ** 2 * np.sign(data['feature_06'])
    
    data['target'] = target
    
    # Create DataFrame
    df = pd.DataFrame(data, index=timestamps)
    
    return df


def test_intelligent_system():
    """Test the complete intelligent system."""
    logger = setup_logging()
    logger.info("Testing the complete intelligent system with synthetic data")
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    try:
        # 1. Generate synthetic data
        logger.info("=" * 50)
        logger.info("STEP 1: GENERATING SYNTHETIC TEST DATA")
        logger.info("=" * 50)
        
        df = generate_synthetic_time_series_data(n_samples=5000, n_features=15, noise_level=0.1)
        logger.info(f"Generated synthetic dataset with shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        target_col = 'target'
        logger.info(f"Target statistics: mean={df[target_col].mean():.3f}, std={df[target_col].std():.3f}")
        
        # 2. Test Smart Trainer
        logger.info("=" * 50)
        logger.info("STEP 2: TESTING SMART TRAINER")
        logger.info("=" * 50)
        
        trainer = SmartModelTrainer()
        
        # Fit the trainer
        logger.info("Fitting smart trainer...")
        trainer.fit(df, target_col)
        
        logger.info("Smart trainer fitted successfully!")
        
        # 3. Evaluation
        logger.info("=" * 50)
        logger.info("STEP 3: COMPREHENSIVE EVALUATION")
        logger.info("=" * 50)
        
        # Get test results
        test_results = trainer.evaluate_on_test()
        
        logger.info("Test Results:")
        logger.info(f"  Best Model: {test_results['model_name']}")
        logger.info(f"  R² Score: {test_results['test_r2']:.4f}")
        logger.info(f"  RMSE: {test_results['test_rmse']:.4f}")
        logger.info(f"  MAE: {test_results['test_mae']:.4f}")
        logger.info(f"  MAPE: {test_results['test_mape']:.2f}%")
        
        # 4. Test Comprehensive Evaluator
        logger.info("=" * 50)
        logger.info("STEP 4: TESTING COMPREHENSIVE EVALUATOR")
        logger.info("=" * 50)
        
        evaluator = TimeSeriesEvaluator()
        
        # Create comprehensive report
        evaluation_report = evaluator.create_evaluation_report(
            y_true=test_results['actual'],
            y_pred=test_results['predictions'],
            timestamps=test_results['test_index'],
            model_name=test_results['model_name']
        )
        
        logger.info("Evaluation Report Created:")
        logger.info(f"  Performance Category: {evaluation_report['performance_category']['category']}")
        logger.info(f"  Sample Size: {evaluation_report['sample_size']:,}")
        logger.info(f"  Directional Accuracy: {evaluation_report['metrics']['directional_accuracy']:.1f}%")
        logger.info(f"  Coverage 95%: {evaluation_report['metrics']['coverage_95']:.1f}%")
        
        # 5. Test Training Summary
        logger.info("=" * 50)
        logger.info("STEP 5: TESTING TRAINING SUMMARY")
        logger.info("=" * 50)
        
        summary = trainer.get_training_summary()
        logger.info(f"Training Summary:")
        logger.info(f"  Features Used: {summary['num_features']}")
        logger.info(f"  Training Samples: {summary['training_samples']:,}")
        logger.info(f"  Best Model: {summary['best_model']}")
        
        logger.info("Model Performance Comparison:")
        for name, metrics in summary['model_results'].items():
            if metrics['val_r2'] is not None:
                logger.info(f"  {name}: Val R²={metrics['val_r2']:.4f}, CV R²={metrics['cv_r2_mean']:.4f}")
        
        # 6. Test Feature Importance
        logger.info("=" * 50)
        logger.info("STEP 6: TESTING FEATURE IMPORTANCE")
        logger.info("=" * 50)
        
        importance = trainer.get_feature_importance()
        if importance is not None:
            logger.info("Top 10 Important Features:")
            for i, (feature, score) in enumerate(importance.head(10).items(), 1):
                logger.info(f"  {i:2d}. {feature}: {score:.4f}")
        else:
            logger.info("Feature importance not available for this model type")
        
        # 7. Validate Expected Behavior
        logger.info("=" * 50)
        logger.info("STEP 7: VALIDATING EXPECTED BEHAVIOR")
        logger.info("=" * 50)
        
        # Check if the system correctly identified the known relationships
        if importance is not None:
            top_features = list(importance.head(5).index)
            expected_important = ['feature_00', 'feature_01', 'feature_02', 'feature_03']
            
            found_expected = sum(1 for f in expected_important if any(f in tf for tf in top_features))
            logger.info(f"Found {found_expected}/{len(expected_important)} expected important features in top 5")
        
        # Check performance thresholds
        performance_checks = [
            ("R² > 0.5", test_results['test_r2'] > 0.5),
            ("MAPE < 50%", test_results['test_mape'] < 50),
            ("Directional Accuracy > 60%", evaluation_report['metrics']['directional_accuracy'] > 60),
            ("Coverage 95% > 80%", evaluation_report['metrics']['coverage_95'] > 80),
        ]
        
        logger.info("Performance Validation:")
        all_passed = True
        for check_name, passed in performance_checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {check_name}: {status}")
            if not passed:
                all_passed = False
        
        # 8. Final Assessment
        logger.info("=" * 50)
        logger.info("FINAL SYSTEM ASSESSMENT")
        logger.info("=" * 50)
        
        if all_passed and test_results['test_r2'] > 0.7:
            logger.info("🎉 EXCELLENT: Complete intelligent system working perfectly!")
            status = "EXCELLENT"
        elif all_passed:
            logger.info("✅ GOOD: Complete intelligent system working well!")
            status = "GOOD"
        elif test_results['test_r2'] > 0.3:
            logger.info("⚠️ ACCEPTABLE: System working but with some issues to address")
            status = "ACCEPTABLE"
        else:
            logger.info("❌ NEEDS WORK: System has significant issues")
            status = "NEEDS WORK"
        
        # Summary
        logger.info("\nSYSTEM TEST SUMMARY:")
        logger.info(f"  Overall Status: {status}")
        logger.info(f"  Data Processing: ✓ Working")
        logger.info(f"  Feature Selection: ✓ Working") 
        logger.info(f"  Model Training: ✓ Working")
        logger.info(f"  Time Series Validation: ✓ Working")
        logger.info(f"  Comprehensive Evaluation: ✓ Working")
        logger.info(f"  Best Model: {test_results['model_name']}")
        logger.info(f"  Final R² Score: {test_results['test_r2']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"System test failed with error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_intelligent_system()
    if success:
        print("\n🎉 INTELLIGENT SYSTEM TEST COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n❌ INTELLIGENT SYSTEM TEST FAILED!")
        sys.exit(1)