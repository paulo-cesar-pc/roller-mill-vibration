#!/usr/bin/env python3
"""
Comprehensive training script for noisy industrial vibration data.
Uses integrated approach with noise reduction, specialized modeling, and alternative formulations.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import get_config
from src.data.data_loader import DataLoader
from src.models.integrated_noisy_trainer import IntegratedNoisyTrainer, create_integrated_config


def setup_logging():
    """Set up comprehensive logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_file = log_dir / f'noisy_training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def analyze_data_characteristics(df: pd.DataFrame, target_col: str, logger: logging.Logger):
    """Analyze and log data characteristics for noisy data assessment."""
    logger.info("=" * 60)
    logger.info("DATA CHARACTERISTICS ANALYSIS")
    logger.info("=" * 60)
    
    # Basic info
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Time range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Time span: {(df.index.max() - df.index.min()).days} days")
    
    # Target analysis
    target_series = df[target_col]
    logger.info(f"\nTarget Variable: {target_col}")
    logger.info(f"  Mean: {target_series.mean():.4f}")
    logger.info(f"  Std: {target_series.std():.4f}")
    logger.info(f"  Min: {target_series.min():.4f}")
    logger.info(f"  Max: {target_series.max():.4f}")
    logger.info(f"  Median: {target_series.median():.4f}")
    logger.info(f"  Skewness: {target_series.skew():.4f}")
    logger.info(f"  Kurtosis: {target_series.kurtosis():.4f}")
    
    # Noise characteristics
    logger.info(f"\nNoise Characteristics:")
    
    # Signal-to-noise ratio estimation
    signal_power = target_series.var()
    noise_estimate = target_series.diff().var() / 2  # Rough noise estimate
    snr = signal_power / noise_estimate if noise_estimate > 0 else np.inf
    logger.info(f"  Estimated SNR: {snr:.2f}")
    
    # Stationarity check (simple)
    rolling_mean = target_series.rolling(window=100).mean()
    rolling_std = target_series.rolling(window=100).std()
    mean_stability = rolling_mean.std() / target_series.mean() if target_series.mean() != 0 else np.inf
    std_stability = rolling_std.std() / target_series.std() if target_series.std() != 0 else np.inf
    
    logger.info(f"  Mean stability (CV): {mean_stability:.4f}")
    logger.info(f"  Std stability (CV): {std_stability:.4f}")
    
    # Autocorrelation
    try:
        autocorr_1 = target_series.autocorr(lag=1)
        autocorr_10 = target_series.autocorr(lag=10)
        logger.info(f"  Autocorr (lag=1): {autocorr_1:.4f}")
        logger.info(f"  Autocorr (lag=10): {autocorr_10:.4f}")
    except Exception as e:
        logger.warning(f"  Could not compute autocorrelation: {e}")
    
    # Feature analysis
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop(target_col)
    logger.info(f"\nFeature Analysis:")
    logger.info(f"  Total features: {len(numeric_features)}")
    logger.info(f"  Missing values: {df[numeric_features].isnull().sum().sum()}")
    
    # High correlation features
    try:
        correlations = df[numeric_features].corrwith(target_series).abs().sort_values(ascending=False)
        top_corr_features = correlations.head(10)
        logger.info(f"  Top correlated features:")
        for feature, corr in top_corr_features.items():
            logger.info(f"    {feature}: {corr:.4f}")
    except Exception as e:
        logger.warning(f"  Could not compute correlations: {e}")


def create_diagnostic_plots(df: pd.DataFrame, target_col: str, output_dir: Path):
    """Create diagnostic plots for noisy data analysis."""
    plots_dir = output_dir / 'diagnostic_plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Target variable analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0, 0].plot(df.index, df[target_col], alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title(f'{target_col} - Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Vibration')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution
    axes[0, 1].hist(df[target_col], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f'{target_col} - Distribution')
    axes[0, 1].set_xlabel('Vibration')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rolling statistics
    rolling_mean = df[target_col].rolling(window=100, min_periods=1).mean()
    rolling_std = df[target_col].rolling(window=100, min_periods=1).std()
    
    axes[1, 0].plot(df.index, rolling_mean, label='Rolling Mean', alpha=0.8)
    axes[1, 0].plot(df.index, rolling_std, label='Rolling Std', alpha=0.8)
    axes[1, 0].set_title('Rolling Statistics (100 samples)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Autocorrelation plot
    try:
        from statsmodels.tsa.stattools import acf
        autocorr = acf(df[target_col].dropna(), nlags=100, fft=True)
        axes[1, 1].plot(range(len(autocorr)), autocorr)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Autocorrelation Function')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
    except ImportError:
        axes[1, 1].text(0.5, 0.5, 'statsmodels not available\nfor autocorrelation plot', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'data_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return plots_dir


def main():
    """Comprehensive noisy data training pipeline."""
    logger = setup_logging()
    logger.info("Starting comprehensive noisy industrial vibration training")
    
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Loaded configuration: {config.project.name} v{config.project.version}")
        
        # Create output directories
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Data Loading
        logger.info("=" * 60)
        logger.info("STEP 1: INTELLIGENT DATA LOADING")
        logger.info("=" * 60)
        
        data_loader = DataLoader()
        df, quality_report = data_loader.load_and_process(testing_mode=True)
        
        # Limit to 50k rows for faster testing
        if len(df) > 50000:
            logger.info(f"Limiting dataset from {len(df)} to 50,000 rows for testing")
            df = df.head(50000)
        
        target_col = config.data.target_column
        logger.info(f"Loaded dataset: {df.shape}")
        logger.info(f"Target column: {target_col}")
        logger.info(f"Data quality score: {quality_report.quality_score:.1f}/100")
        
        # 2. Data Characteristics Analysis
        analyze_data_characteristics(df, target_col, logger)
        
        # 3. Create Diagnostic Plots
        logger.info("Creating diagnostic plots...")
        plots_dir = create_diagnostic_plots(df, target_col, output_dir)
        logger.info(f"Diagnostic plots saved to: {plots_dir}")
        
        # 4. Integrated Noisy Training
        logger.info("=" * 60)
        logger.info("STEP 2: INTEGRATED NOISY DATA TRAINING")
        logger.info("=" * 60)
        
        # Create integrated configuration
        noise_config = create_integrated_config(
            target_frequency='5min',
            focus_on_robustness=True,
            include_all_formulations=True
        )
        
        # Initialize and train integrated trainer
        integrated_trainer = IntegratedNoisyTrainer(noise_config)
        
        # Get feature columns (exclude target)
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Train the integrated pipeline
        training_results = integrated_trainer.fit(df, target_col, feature_cols)
        
        # 5. Results Analysis
        logger.info("=" * 60)
        logger.info("STEP 3: COMPREHENSIVE RESULTS ANALYSIS")
        logger.info("=" * 60)
        
        # Training summary
        training_summary = integrated_trainer.get_training_summary()
        
        logger.info("TRAINING SUMMARY:")
        logger.info(f"  Original data shape: {training_summary['data_summary'].get('original_shape', 'N/A')}")
        logger.info(f"  Processed data shape: {training_summary['data_summary'].get('processed_shape', 'N/A')}")
        logger.info(f"  Final feature count: {training_summary.get('feature_count', 0)}")
        
        # Model performance
        model_perf = training_summary.get('model_performance', {})
        if model_perf:
            logger.info("BEST MODEL PERFORMANCE:")
            logger.info(f"  Cross-validation R²: {model_perf.get('cv_r2_mean', 0):.4f} ± {model_perf.get('cv_r2_std', 0):.4f}")
            logger.info(f"  Cross-validation RMSE: {model_perf.get('cv_rmse_mean', 0):.4f}")
            logger.info(f"  Cross-validation MAE: {model_perf.get('cv_mae_mean', 0):.4f}")
        
        # Feature importance
        feature_importance = integrated_trainer.get_feature_importance()
        if not feature_importance.empty:
            logger.info("TOP 10 IMPORTANT FEATURES:")
            for idx, row in feature_importance.head(10).iterrows():
                mean_val = float(row['mean']) if hasattr(row['mean'], '__float__') else row['mean']
                std_val = float(row['std']) if hasattr(row['std'], '__float__') else row['std']
                logger.info(f"  {row['feature']}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Ensemble model details
        if 'ensemble_models' in training_results:
            ensemble_results = training_results['ensemble_models']
            logger.info("ENSEMBLE MODEL RESULTS:")
            for model_name, results in ensemble_results.items():
                if isinstance(results, dict) and 'error' not in results:
                    r2_mean = results.get('r2_mean', 0)
                    robustness = results.get('robustness', 0)
                    logger.info(f"  {model_name}: R²={r2_mean:.4f}, Robustness={robustness:.4f}")
        
        # Alternative formulations
        if 'alternative_formulations' in training_results:
            alt_results = training_results['alternative_formulations']
            logger.info("ALTERNATIVE FORMULATIONS:")
            
            # Classification results
            if 'classification' in alt_results:
                class_results = alt_results['classification']
                logger.info("  Classification Models:")
                for model_name, results in class_results.items():
                    if isinstance(results, dict) and 'cv_accuracy_mean' in results:
                        acc = results['cv_accuracy_mean']
                        logger.info(f"    {model_name}: {acc:.4f} accuracy")
            
            # Anomaly detection
            if 'anomaly_detection' in alt_results:
                anomaly_results = alt_results['anomaly_detection']
                logger.info("  Anomaly Detection:")
                for model_name, results in anomaly_results.items():
                    if isinstance(results, dict) and 'anomaly_rate' in results:
                        rate = results['anomaly_rate']
                        logger.info(f"    {model_name}: {rate:.4f} anomaly rate")
        
        # 6. Practical Recommendations
        logger.info("=" * 60)
        logger.info("PRACTICAL RECOMMENDATIONS")
        logger.info("=" * 60)
        
        # Performance assessment
        best_r2 = model_perf.get('cv_r2_mean', 0)
        if best_r2 > 0.6:
            logger.info("🎉 EXCELLENT: Model shows strong predictive capability")
            logger.info("   Recommend: Deploy for predictive maintenance")
        elif best_r2 > 0.3:
            logger.info("✅ GOOD: Model shows reasonable predictive capability")
            logger.info("   Recommend: Use for trend monitoring and alerts")
        elif best_r2 > 0.1:
            logger.info("⚠️ ACCEPTABLE: Model performs better than baseline")
            logger.info("   Recommend: Focus on anomaly detection and classification")
        else:
            logger.info("❌ POOR: Model struggles with prediction")
            logger.info("   Recommend: Focus on threshold monitoring and process optimization")
        
        # Data quality recommendations
        if quality_report.quality_score < 70:
            logger.info("📊 DATA QUALITY RECOMMENDATIONS:")
            logger.info("   - Consider additional data preprocessing")
            logger.info("   - Investigate sensor calibration")
            logger.info("   - Review data collection frequency")
        
        # Alternative approach recommendations
        logger.info("🔄 ALTERNATIVE APPROACHES:")
        logger.info("   - Classification: Use for operational state detection")
        logger.info("   - Anomaly Detection: Implement for early warning systems")
        logger.info("   - Threshold Monitoring: Deploy for immediate alerts")
        
        # 7. Save Final Results
        logger.info("=" * 60)
        logger.info("FINAL RESULTS SAVED")
        logger.info("=" * 60)
        
        # Results are automatically saved by the integrated trainer
        logger.info(f"All results saved to: {output_dir}")
        logger.info("Files generated:")
        logger.info(f"  - Models: {output_dir}/models/")
        logger.info(f"  - Plots: {output_dir}/plots/")
        logger.info(f"  - Reports: {output_dir}/reports/")
        logger.info(f"  - Diagnostics: {plots_dir}")
        
        # Success summary
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        logger.info(f"✅ Best Model R²: {best_r2:.4f}")
        logger.info(f"✅ Feature Engineering: {df.shape[1]} → {training_summary.get('feature_count', 0)} features")
        logger.info(f"✅ Multiple Approaches: Regression, Classification, Anomaly Detection")
        logger.info(f"✅ Robust to Noise: Specialized algorithms and preprocessing")
        
        return 0
        
    except Exception as e:
        logger.error(f"Comprehensive training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)