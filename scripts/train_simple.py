#!/usr/bin/env python3
"""
Simplified training script for initial testing.
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
from src.features.feature_engineer import create_default_pipeline
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import evaluate_model_comprehensive


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def main():
    """Simplified training pipeline."""
    logger = setup_logging()
    logger.info("Starting simplified roller mill vibration prediction training")
    
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Loaded configuration: {config.project.name} v{config.project.version}")
        
        # 1. Data Loading (use existing processed data if available)
        logger.info("=" * 50)
        logger.info("STEP 1: DATA LOADING")
        logger.info("=" * 50)
        
        processed_data_path = Path("data/processed/processed_data.csv")
        
        if processed_data_path.exists():
            logger.info(f"Loading existing processed data from {processed_data_path}")
            df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        else:
            logger.info("Processed data not found. Please run the full training script first.")
            return 1
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # 2. Feature Engineering (simplified)
        logger.info("=" * 50)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 50)
        
        target_col = config.data.target_column
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            return 1
        
        # For simplicity, use only the most correlated features
        feature_cols = [
            'CM2_PV_VRM01_DIFF_PRESSURE',
            'CM2_PV_VRM01_OUT_ PRESS', 
            'CM2_PV_WI01_WATER_INJECTION',
            'CM2_PV_VRM01_POWER',
            'CM2_PV_CLA01_SPEED',
            'CM2_PV_BE01_CURRENT',
            'CM2_PV_BF01_PRESSURE',
            'CM2_PV_FN01_POWER',
            'CM2_PV_VRM01_IN_PRESS'
        ]
        
        # Keep only available columns
        available_features = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_features)} features: {available_features}")
        
        # Simple feature engineering: just add time features
        X = df[available_features].copy()
        y = df[target_col].copy()
        
        # Add basic time features
        X['hour_sin'] = np.sin(2 * np.pi * X.index.hour / 24)
        X['hour_cos'] = np.cos(2 * np.pi * X.index.hour / 24)
        X['day_of_week_sin'] = np.sin(2 * np.pi * X.index.dayofweek / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X.index.dayofweek / 7)
        
        # Remove NaN values
        valid_mask = ~X.isnull().any(axis=1) & ~y.isnull()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"After cleaning: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        
        # 3. Data Splitting
        logger.info("=" * 50)
        logger.info("STEP 3: DATA SPLITTING")
        logger.info("=" * 50)
        
        # Simple time-based split
        n_total = len(X_clean)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        X_train = X_clean.iloc[:n_train]
        X_val = X_clean.iloc[n_train:n_train+n_val] 
        X_test = X_clean.iloc[n_train+n_val:]
        
        y_train = y_clean.iloc[:n_train]
        y_val = y_clean.iloc[n_train:n_train+n_val]
        y_test = y_clean.iloc[n_train+n_val:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # 4. Model Training (simplified)
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("=" * 50)
        
        # Simple models without hyperparameter optimization
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        import xgboost as xgb
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_mae = mean_absolute_error(y_val, val_pred)
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae
            }
            
            logger.info(f"{name} - Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
        
        # 5. Select best model and evaluate on test set
        logger.info("=" * 50)
        logger.info("STEP 5: FINAL EVALUATION")
        logger.info("=" * 50)
        
        # Find best model by validation R²
        best_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
        best_model = results[best_name]['model']
        
        logger.info(f"Best model: {best_name}")
        
        # Test evaluation
        test_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        logger.info(f"Test Results - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        # 6. Save results
        logger.info("=" * 50)
        logger.info("SAVING RESULTS")
        logger.info("=" * 50)
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("plots").mkdir(exist_ok=True)
        
        # Save best model
        import joblib
        model_path = f"models/best_model_{best_name.replace(' ', '_')}.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to: {model_path}")
        
        # Generate basic plots
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Predictions vs Actual
        plt.subplot(1, 3, 1)
        plt.scatter(y_test, test_pred, alpha=0.6, s=1)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{best_name}: Test Predictions vs Actual\nR² = {test_r2:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Time series comparison (sample)
        plt.subplot(1, 3, 2)
        sample_size = min(1000, len(X_test))
        sample_idx = np.random.choice(len(X_test), sample_size, replace=False)
        sample_idx = np.sort(sample_idx)
        
        plt.plot(X_test.iloc[sample_idx].index, y_test.iloc[sample_idx], 
                label='Actual', alpha=0.7, linewidth=1)
        plt.plot(X_test.iloc[sample_idx].index, test_pred[sample_idx], 
                label='Predicted', alpha=0.7, linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Vibration')
        plt.title('Time Series Comparison (Sample)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot 3: Residuals
        plt.subplot(1, 3, 3)
        residuals = y_test - test_pred
        plt.scatter(test_pred, residuals, alpha=0.6, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"plots/{best_name.replace(' ', '_')}_evaluation.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Evaluation plots saved to: {plot_path}")
        plt.show()
        
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        
        # Print summary
        logger.info("SUMMARY:")
        logger.info(f"  Dataset: {len(X_clean):,} samples, {X_clean.shape[1]} features")
        logger.info(f"  Best model: {best_name}")
        logger.info(f"  Test R²: {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.4f}")
        logger.info(f"  Test MAE: {test_mae:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)