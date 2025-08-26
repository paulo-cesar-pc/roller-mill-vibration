#!/usr/bin/env python3
"""
Intelligent training script using smart feature selection and proper time series validation.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import get_config
from src.models.smart_trainer import SmartModelTrainer
from src.data.data_loader import DataLoader
from src.evaluation.time_series_evaluator import TimeSeriesEvaluator


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_data(logger):
    """Load and validate data."""
    logger.info("Loading data...")
    
    # Try multiple data sources
    data_paths = [
        Path("data/processed/processed_data.csv"),
        Path("data/raw/roller_mill_data.csv"),
        Path("full_data/roller_mill_data.csv")
    ]
    
    df = None
    for path in data_paths:
        if path.exists():
            logger.info(f"Loading data from {path}")
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                logger.info(f"Successfully loaded data with shape: {df.shape}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue
    
    if df is None:
        # Try using DataLoader with testing mode enabled
        logger.info("Attempting to load data using DataLoader with testing mode...")
        try:
            data_loader = DataLoader()
            df, quality_report = data_loader.load_and_process(testing_mode=True)
            logger.info(f"DataLoader successful: {df.shape}")
            logger.info(f"Data quality score: {quality_report.quality_score:.1f}/100")
        except Exception as e:
            logger.error(f"DataLoader failed: {e}")
            raise FileNotFoundError("No data file found. Please check data paths.")
    
    return df


def create_comprehensive_evaluation(trainer, results, output_dir, model_name):
    """Create comprehensive evaluation using the TimeSeriesEvaluator."""
    
    # Initialize the evaluator
    evaluator = TimeSeriesEvaluator()
    
    # Create comprehensive evaluation report
    evaluation_report = evaluator.create_evaluation_report(
        y_true=results['actual'],
        y_pred=results['predictions'],
        timestamps=results['test_index'],
        model_name=model_name
    )
    
    # Create comprehensive plots
    fig = evaluator.create_evaluation_plots(
        y_true=results['actual'],
        y_pred=results['predictions'],
        timestamps=results['test_index'],
        model_name=model_name,
        figsize=(20, 15)
    )
    
    # Save comprehensive plot
    plot_path = output_dir / f"comprehensive_evaluation_{model_name.replace(' ', '_')}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return evaluation_report, plot_path


def main():
    """Intelligent training pipeline."""
    logger = setup_logging()
    logger.info("Starting intelligent roller mill vibration prediction training")
    
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Loaded configuration: {config.project.name} v{config.project.version}")
        
        # Create output directories
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        (output_dir / "models").mkdir(exist_ok=True)
        (output_dir / "plots").mkdir(exist_ok=True)
        
        # 1. Data Loading
        logger.info("=" * 60)
        logger.info("STEP 1: INTELLIGENT DATA LOADING")
        logger.info("=" * 60)
        
        df = load_data(logger)
        
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Target column: {config.data.target_column}")
        
        # Quick data quality check
        target_col = config.data.target_column
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            logger.info(f"Available columns: {list(df.columns)}")
            return 1
        
        # Check target distribution
        target_stats = df[target_col].describe()
        logger.info(f"Target statistics:\n{target_stats}")
        
        # 2. Smart Training
        logger.info("=" * 60)
        logger.info("STEP 2: INTELLIGENT MODEL TRAINING")
        logger.info("=" * 60)
        
        # Initialize smart trainer
        trainer = SmartModelTrainer()
        
        # Fit the smart trainer (this does everything intelligently)
        trainer.fit(df, target_col)
        
        # 3. Evaluation
        logger.info("=" * 60)
        logger.info("STEP 3: COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)
        
        # Evaluate on test set
        test_results = trainer.evaluate_on_test()
        
        # Print detailed results
        logger.info("=" * 40)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 40)
        
        logger.info(f"Best Model: {test_results['model_name']}")
        logger.info(f"Test R²: {test_results['test_r2']:.4f}")
        logger.info(f"Test RMSE: {test_results['test_rmse']:.4f}")
        logger.info(f"Test MAE: {test_results['test_mae']:.4f}")
        logger.info(f"Test MAPE: {test_results['test_mape']:.2f}%")
        
        # Training summary
        summary = trainer.get_training_summary()
        if summary:
            logger.info("=" * 40)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 40)
            
            logger.info(f"Features used: {summary['num_features']}")
            logger.info(f"Training samples: {summary['training_samples']:,}")
            logger.info(f"Validation samples: {summary['validation_samples']:,}")
            logger.info(f"Test samples: {summary['test_samples']:,}")
            
            logger.info("\nModel Performance Comparison:")
            for name, metrics in summary['model_results'].items():
                logger.info(f"  {name}: Val R²={metrics['val_r2']:.4f}, "
                           f"CV R²={metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}")
        
        # Feature importance
        importance = trainer.get_feature_importance()
        if importance is not None:
            logger.info("=" * 40)
            logger.info("TOP 10 IMPORTANT FEATURES")
            logger.info("=" * 40)
            for i, (feature, score) in enumerate(importance.head(10).items(), 1):
                logger.info(f"{i:2d}. {feature}: {score:.4f}")
        
        # 4. Save Results
        logger.info("=" * 60)
        logger.info("STEP 4: SAVING RESULTS")
        logger.info("=" * 60)
        
        # Save model
        model_path = output_dir / "models" / f"intelligent_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_model(str(model_path))
        logger.info(f"Model saved to: {model_path}")
        
        # Create comprehensive evaluation
        evaluation_report, plot_path = create_comprehensive_evaluation(
            trainer, test_results, output_dir / "plots", test_results['model_name']
        )
        logger.info(f"Comprehensive evaluation plots saved to: {plot_path}")
        
        # Save comprehensive evaluation report
        results_path = output_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_path, 'w') as f:
            f.write("COMPREHENSIVE INTELLIGENT TRAINING EVALUATION\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic info
            f.write(f"Model: {evaluation_report['model_name']}\n")
            f.write(f"Evaluation Time: {evaluation_report['evaluation_timestamp']}\n")
            f.write(f"Sample Size: {evaluation_report['sample_size']:,}\n")
            f.write(f"Performance Category: {evaluation_report['performance_category']['category']}\n")
            f.write(f"Description: {evaluation_report['performance_category']['description']}\n\n")
            
            # Comprehensive metrics
            metrics = evaluation_report['metrics']
            f.write("COMPREHENSIVE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"R² Score: {metrics['r2_score']:.6f}\n")
            f.write(f"RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"MAE: {metrics['mae']:.6f}\n")
            f.write(f"Median AE: {metrics['medae']:.6f}\n")
            f.write(f"MAPE: {metrics['mape']:.4f}%\n")
            f.write(f"SMAPE: {metrics['smape']:.4f}%\n")
            f.write(f"MASE: {metrics['mase']:.6f}\n")
            f.write(f"Explained Variance: {metrics['explained_variance']:.6f}\n")
            f.write(f"Directional Accuracy: {metrics['directional_accuracy']:.2f}%\n")
            f.write(f"Coverage 68%: {metrics['coverage_68']:.2f}%\n")
            f.write(f"Coverage 95%: {metrics['coverage_95']:.2f}%\n")
            f.write(f"Accuracy ±1σ: {metrics['accuracy_1std']:.2f}%\n")
            f.write(f"Accuracy ±2σ: {metrics['accuracy_2std']:.2f}%\n\n")
            
            # Residual analysis
            residual = evaluation_report['residual_analysis']
            f.write("RESIDUAL ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean: {residual['mean']:.6f}\n")
            f.write(f"Std Dev: {residual['std']:.6f}\n")
            f.write(f"Skewness: {residual['skewness']:.6f}\n")
            f.write(f"Kurtosis: {residual['kurtosis']:.6f}\n")
            f.write(f"Normality (Shapiro): {residual['normality_test']['is_normal']}\n")
            
            if residual['homoscedasticity_test']['is_homoscedastic'] is not None:
                f.write(f"Homoscedasticity: {residual['homoscedasticity_test']['is_homoscedastic']}\n")
            if residual['autocorrelation_test']['is_uncorrelated'] is not None:
                f.write(f"No Autocorrelation: {residual['autocorrelation_test']['is_uncorrelated']}\n")
            f.write("\n")
            
            # Training summary
            if summary:
                f.write("TRAINING SUMMARY\n")
                f.write("-" * 20 + "\n")
                f.write(f"Features used: {summary['num_features']}\n")
                f.write(f"Training samples: {summary['training_samples']:,}\n")
                f.write(f"Validation samples: {summary['validation_samples']:,}\n")
                f.write(f"Test samples: {summary['test_samples']:,}\n\n")
                
                f.write("Model Performance Comparison:\n")
                for name, metrics in summary['model_results'].items():
                    f.write(f"  {name}: Val R²={metrics['val_r2']:.6f}, "
                           f"CV R²={metrics['cv_r2_mean']:.6f}±{metrics['cv_r2_std']:.6f}\n")
                f.write("\n")
            
            # Feature importance
            if importance is not None:
                f.write("TOP 20 IMPORTANT FEATURES\n")
                f.write("-" * 25 + "\n")
                for i, (feature, score) in enumerate(importance.head(20).items(), 1):
                    f.write(f"{i:2d}. {feature}: {score:.6f}\n")
                f.write("\n")
            
            # Summary text from evaluator
            f.write("EVALUATION SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(evaluation_report['summary'])
        
        logger.info(f"Comprehensive evaluation report saved to: {results_path}")
        
        # Success message
        logger.info("=" * 60)
        logger.info("INTELLIGENT TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        # Final summary
        if test_results['test_r2'] > 0.5:
            logger.info("🎉 EXCELLENT: Model achieved good predictive performance!")
        elif test_results['test_r2'] > 0.2:
            logger.info("✅ GOOD: Model shows reasonable predictive capability.")
        elif test_results['test_r2'] > 0:
            logger.info("⚠️  ACCEPTABLE: Model performs better than baseline but has room for improvement.")
        else:
            logger.info("❌ POOR: Model performance is below baseline. Data quality issues may exist.")
        
        logger.info(f"Final Test R²: {test_results['test_r2']:.4f}")
        logger.info(f"Model: {test_results['model_name']}")
        logger.info(f"Features: {summary['num_features'] if summary else 'Unknown'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Intelligent training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)