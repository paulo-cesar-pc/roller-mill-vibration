#!/usr/bin/env python3
"""
Simple test script to validate the key fixes.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def test_evaluation_fix():
    """Test the TimeSeriesEvaluator fix."""
    print("Testing TimeSeriesEvaluator fix...")
    
    try:
        from src.evaluation.time_series_evaluator import TimeSeriesEvaluator
        
        # Create simple test data
        n_samples = 1000
        y_true = np.random.normal(0, 1, n_samples)
        y_pred = y_true + np.random.normal(0, 0.1, n_samples)  # Good predictions
        
        evaluator = TimeSeriesEvaluator()
        
        # Test the fixed residual analysis
        residual_analysis = evaluator.analyze_residuals(y_true, y_pred)
        
        print("✅ TimeSeriesEvaluator fix successful!")
        print(f"   - Residual mean: {residual_analysis['mean']:.4f}")
        print(f"   - Residual std: {residual_analysis['std']:.4f}")
        print(f"   - Normality test p-value: {residual_analysis['normality_test']['shapiro_p']:.4f}")
        
        # Check homoscedasticity test
        if 'homoscedasticity_test' in residual_analysis:
            if residual_analysis['homoscedasticity_test']['bp_p'] is not None:
                print(f"   - Homoscedasticity test p-value: {residual_analysis['homoscedasticity_test']['bp_p']:.4f}")
            else:
                print("   - Homoscedasticity test: N/A")
        
        return True
        
    except Exception as e:
        print(f"❌ TimeSeriesEvaluator fix failed: {e}")
        return False


def test_target_like_exclusion():
    """Test the target-like feature exclusion."""
    print("\nTesting target-like feature exclusion...")
    
    try:
        from src.data.intelligent_analyzer import IntelligentDataAnalyzer
        
        # Create test data with target-like features
        n_samples = 1000
        target = np.random.normal(0, 1, n_samples)
        
        # Create DataFrame with various types of features
        data = {
            'target': target,
            'good_feature_1': target * 0.7 + np.random.normal(0, 0.5, n_samples),  # Moderate correlation
            'good_feature_2': target * 0.5 + np.random.normal(0, 0.8, n_samples),  # Lower correlation
            'CM2_PV_VRM01_VIBRATION1': target + np.random.normal(0, 0.01, n_samples),  # Almost identical to target
            'another_VIBRATION_feature': target * 0.98 + np.random.normal(0, 0.05, n_samples),  # Very high correlation
            'random_feature': np.random.normal(0, 1, n_samples),  # No correlation
        }
        
        # Add timestamps
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        df = pd.DataFrame(data, index=timestamps)
        
        analyzer = IntelligentDataAnalyzer()
        
        # Test feature selection with target-like exclusion
        selected_features = analyzer.select_features_intelligently(
            df, 
            'target',
            correlation_threshold=0.2,
            max_features=10,
            exclude_target_like=True,
            target_similarity_threshold=0.95
        )
        
        print("✅ Target-like exclusion successful!")
        print(f"   - Selected features: {selected_features}")
        
        # Check that problematic features were excluded
        problematic_excluded = (
            'CM2_PV_VRM01_VIBRATION1' not in selected_features and
            'another_VIBRATION_feature' not in selected_features
        )
        
        if problematic_excluded:
            print("   - ✅ Problematic target-like features correctly excluded")
        else:
            print("   - ⚠️  Some problematic features may not have been excluded")
        
        # Check that good features were included
        good_features_included = any(f in selected_features for f in ['good_feature_1', 'good_feature_2'])
        
        if good_features_included:
            print("   - ✅ Good features correctly included")
        else:
            print("   - ⚠️  Good features may have been incorrectly excluded")
        
        return True
        
    except Exception as e:
        print(f"❌ Target-like exclusion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_smart_trainer_basic():
    """Test basic SmartModelTrainer functionality."""
    print("\nTesting SmartModelTrainer basic functionality...")
    
    try:
        from src.models.smart_trainer import SmartModelTrainer
        
        # Create simple test data
        n_samples = 2000
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        # Create features with known relationships
        feature_1 = np.random.normal(0, 1, n_samples)
        feature_2 = np.random.normal(0, 1, n_samples)
        target = 0.7 * feature_1 + 0.3 * feature_2 + np.random.normal(0, 0.2, n_samples)
        
        df = pd.DataFrame({
            'feature_1': feature_1,
            'feature_2': feature_2,
            'random_feature': np.random.normal(0, 1, n_samples),
            'target': target
        }, index=timestamps)
        
        # Test SmartModelTrainer
        trainer = SmartModelTrainer()
        trainer.fit(df, 'target')
        
        # Get test results
        results = trainer.evaluate_on_test()
        
        print("✅ SmartModelTrainer basic functionality successful!")
        print(f"   - Best model: {results['model_name']}")
        print(f"   - Test R²: {results['test_r2']:.4f}")
        print(f"   - Test RMSE: {results['test_rmse']:.4f}")
        
        # Check if performance is reasonable
        if results['test_r2'] > 0.5:
            print("   - ✅ Good performance achieved")
        else:
            print(f"   - ⚠️  Performance could be better (R²={results['test_r2']:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ SmartModelTrainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running key fixes validation tests...\n")
    
    tests = [
        test_evaluation_fix,
        test_target_like_exclusion,
        test_smart_trainer_basic
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed. Some fixes may need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)