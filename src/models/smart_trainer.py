"""
Smart model trainer with intelligent feature selection and proper time series validation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
from pathlib import Path
import joblib

from src.data.intelligent_analyzer import IntelligentDataAnalyzer
from src.data.time_series_validator import TimeSeriesValidator, WalkForwardValidator
from src.features.smart_feature_engineer import SmartFeatureEngineer
from config.settings import get_config


class SmartModelTrainer:
    """Smart model trainer with intelligent feature selection and proper validation."""
    
    def __init__(self):
        """Initialize the smart model trainer."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        self.analyzer = IntelligentDataAnalyzer()
        self.validator = TimeSeriesValidator()
        self.feature_engineer = SmartFeatureEngineer()
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.scaler = None
        self.is_fitted = False
        
        # Training results
        self.training_results = {}
        self.cv_results = {}
        
    def _get_model_candidates(self) -> Dict[str, Any]:
        """Get candidate models with proper regularization."""
        models = {
            # Linear models with regularization
            'Ridge': Pipeline([
                ('scaler', RobustScaler()),
                ('model', Ridge(alpha=1.0, random_state=42))
            ]),
            
            'Lasso': Pipeline([
                ('scaler', RobustScaler()),
                ('model', Lasso(alpha=0.1, random_state=42))
            ]),
            
            'ElasticNet': Pipeline([
                ('scaler', RobustScaler()),
                ('model', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42))
            ]),
            
            # Tree-based models with reduced complexity
            'Random Forest': RandomForestRegressor(
                n_estimators=50,  # Reduced from default
                max_depth=10,     # Limit tree depth
                min_samples_split=10,  # Increase minimum samples
                min_samples_leaf=5,    # Increase minimum leaf samples
                max_features='sqrt',   # Use subset of features
                random_state=42
            ),
            
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,  # Use subset of samples
                random_state=42
            ),
            
            # XGBoost with regularization
            'XGBoost': xgb.XGBRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0, # L2 regularization
                random_state=42
            )
        }
        
        # Add LightGBM if available
        try:
            models['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1
            )
        except Exception:
            self.logger.warning("LightGBM not available, skipping")
        
        return models
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data using intelligent feature selection and engineering."""
        self.logger.info("Preparing data with intelligent feature selection")
        
        # Step 1: Intelligent feature selection
        self.logger.info("Analyzing correlations for feature selection")
        selected_features = self.analyzer.select_features_intelligently(
            df, 
            target_column,
            correlation_threshold=0.15,  # Lower threshold for more features
            max_features=20
        )
        
        if not selected_features:
            raise ValueError("No significant features found. Check data quality and target correlation.")
        
        self.logger.info(f"Selected {len(selected_features)} base features")
        
        # Step 2: Smart feature engineering
        self.logger.info("Applying smart feature engineering")
        
        # Create feature set with selected features and target
        feature_df = df[selected_features + [target_column]].copy()
        
        # Fit and transform with smart feature engineering
        X_engineered = self.feature_engineer.fit_transform(feature_df, target_column)
        y = feature_df[target_column].copy()
        
        # Align target with engineered features (handle any dropped rows)
        common_index = X_engineered.index.intersection(y.index)
        X_engineered = X_engineered.loc[common_index]
        y = y.loc[common_index]
        
        self.logger.info(f"Feature engineering complete: {len(selected_features)} -> {X_engineered.shape[1]} features")
        
        # Step 3: Handle missing values
        initial_samples = len(X_engineered)
        
        # Remove rows with any NaN values
        valid_mask = ~X_engineered.isnull().any(axis=1) & ~y.isnull()
        X_clean = X_engineered[valid_mask]
        y_clean = y[valid_mask]
        
        removed_samples = initial_samples - len(X_clean)
        if removed_samples > 0:
            self.logger.info(f"Removed {removed_samples} samples with missing values")
        
        if len(X_clean) < 1000:
            raise ValueError(f"Too few samples after cleaning: {len(X_clean)}. Check data quality.")
        
        # Store feature names for later use
        self.feature_names = list(X_clean.columns)
        
        self.logger.info(f"Final dataset: {len(X_clean)} samples, {len(self.feature_names)} features")
        
        return X_clean, y_clean
        
    def create_time_series_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        gap_days: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Create proper time series splits to prevent data leakage."""
        self.logger.info("Creating time series splits")
        
        # Combine X and y for splitting
        data_combined = X.copy()
        data_combined['target'] = y
        
        # Auto-determine gap_days based on dataset size and time span
        if gap_days is None:
            time_span = X.index.max() - X.index.min()
            total_days = time_span.total_seconds() / (24 * 3600)
            
            self.logger.info(f"Dataset spans {total_days:.1f} days with {len(X)} samples")
            
            if total_days < 7:  # Less than a week of data
                gap_days = 0  # No gaps for small datasets
                self.logger.info("Using no gaps due to small dataset timespan")
            elif total_days < 30:  # Less than a month
                gap_days = 1  # Small gap
                self.logger.info("Using 1-day gap for medium dataset timespan")
            else:
                gap_days = 1  # Standard gap
                self.logger.info("Using 1-day gap for large dataset timespan")
        
        # Use time series validator for proper splits
        train_df, val_df, test_df = self.validator.create_temporal_splits(
            data_combined,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=1.0 - train_ratio - val_ratio,
            gap_days=gap_days
        )
        
        # Separate features and target
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        
        X_val = val_df.drop('target', axis=1)
        y_val = val_df['target']
        
        X_test = test_df.drop('target', axis=1) 
        y_test = test_df['target']
        
        # Validate splits
        self.logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        self.logger.info(f"Train date range: {X_train.index.min()} to {X_train.index.max()}")
        self.logger.info(f"Val date range: {X_val.index.min()} to {X_val.index.max()}")
        self.logger.info(f"Test date range: {X_test.index.min()} to {X_test.index.max()}")
        
        # Check for data leakage
        from src.data.time_series_validator import TimeSeriesLeakageDetector
        leakage_detector = TimeSeriesLeakageDetector()
        
        self.logger.info("Checking for temporal leakage in train/validation split...")
        temporal_leakage = leakage_detector.detect_temporal_leakage(
            X_train.index, X_val.index
        )
        
        self.logger.info(f"Temporal leakage results: {temporal_leakage}")
        
        if not temporal_leakage['is_valid_temporal_split']:
            self.logger.error("Temporal data leakage detected in train/val split!")
            self.logger.error("This means training data timestamps overlap with or come after validation timestamps")
            raise ValueError("Invalid temporal split - data leakage detected")
        
        # Also check val/test split
        self.logger.info("Checking for temporal leakage in validation/test split...")
        val_test_leakage = leakage_detector.detect_temporal_leakage(
            X_val.index, X_test.index
        )
        
        if not val_test_leakage['is_valid_temporal_split']:
            self.logger.error("Temporal data leakage detected in validation/test split!")
            self.logger.error("This means validation data timestamps overlap with or come after test timestamps")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def train_and_validate_models(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        cv_folds: int = 3
    ) -> Dict[str, Any]:
        """Train and validate models with proper time series cross-validation."""
        self.logger.info("Training and validating models")
        
        models = self._get_model_candidates()
        results = {}
        
        # Set up time series cross-validation
        cv_splitter = WalkForwardValidator(n_splits=cv_folds, gap=1)
        
        for name, model in models.items():
            self.logger.info(f"Training {name}")
            
            try:
                # Fit on training set
                model.fit(X_train, y_train)
                
                # Predictions
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                # Metrics
                train_r2 = r2_score(y_train, train_pred)
                val_r2 = r2_score(y_val, val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                val_mae = mean_absolute_error(y_val, val_pred)
                
                # Cross-validation on training set
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_splitter,
                    scoring='r2',
                    n_jobs=-1
                )
                
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'cv_r2_mean': cv_mean,
                    'cv_r2_std': cv_std,
                    'cv_scores': cv_scores
                }
                
                self.logger.info(f"{name}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, CV R²={cv_mean:.4f}±{cv_std:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                continue
        
        return results
        
    def select_best_model(self, results: Dict[str, Any]) -> Tuple[str, Any]:
        """Select the best model based on validation performance."""
        if not results:
            raise ValueError("No models successfully trained")
        
        # Select based on validation R² (but ensure it's positive)
        valid_models = {
            name: res for name, res in results.items()
            if res['val_r2'] > 0 and res['cv_r2_mean'] > 0
        }
        
        if not valid_models:
            self.logger.warning("No models with positive R². Selecting least negative.")
            valid_models = results
        
        # Select model with best validation R² and reasonable CV performance
        best_name = max(valid_models.keys(), 
                       key=lambda x: valid_models[x]['val_r2'])
        
        best_model = valid_models[best_name]['model']
        
        self.logger.info(f"Best model selected: {best_name}")
        return best_name, best_model
        
    def fit(
        self,
        df: pd.DataFrame,
        target_column: str
    ) -> 'SmartModelTrainer':
        """Fit the smart model trainer."""
        self.logger.info("Starting smart model training")
        
        try:
            # Step 1: Prepare data
            X, y = self.prepare_data(df, target_column)
            
            # Step 2: Create time series splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.create_time_series_splits(X, y)
            
            # Step 3: Train and validate models
            self.training_results = self.train_and_validate_models(X_train, X_val, y_train, y_val)
            
            # Step 4: Select best model
            self.best_model_name, self.best_model = self.select_best_model(self.training_results)
            
            # Store datasets for evaluation
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test
            self.y_train = y_train
            self.y_val = y_val
            self.y_test = y_test
            
            self.is_fitted = True
            self.logger.info("Smart model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Smart model training failed: {e}")
            raise
            
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the best model."""
        if not self.is_fitted:
            raise ValueError("SmartModelTrainer must be fitted before making predictions")
        
        # Ensure features match training
        if list(X.columns) != self.feature_names:
            # Try to reorder or select matching features
            available_features = [f for f in self.feature_names if f in X.columns]
            if len(available_features) != len(self.feature_names):
                raise ValueError(f"Feature mismatch. Expected: {self.feature_names}, Got: {list(X.columns)}")
            X = X[self.feature_names]
        
        return self.best_model.predict(X)
        
    def evaluate_on_test(self) -> Dict[str, Any]:
        """Evaluate the best model on test set."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Test predictions
        test_pred = self.best_model.predict(self.X_test)
        
        # Calculate metrics
        test_r2 = r2_score(self.y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, test_pred))
        test_mae = mean_absolute_error(self.y_test, test_pred)
        
        # Additional metrics
        test_mape = np.mean(np.abs((self.y_test - test_pred) / self.y_test)) * 100
        residuals = self.y_test - test_pred
        
        results = {
            'model_name': self.best_model_name,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'predictions': test_pred,
            'actual': self.y_test.values,
            'test_index': self.X_test.index
        }
        
        self.logger.info(f"Test Results - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        return results
        
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance from the best model."""
        if not self.is_fitted:
            return None
            
        model = self.best_model
        
        # Extract the actual model from pipeline if needed
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['model']
        else:
            actual_model = model
        
        # Get importance based on model type
        if hasattr(actual_model, 'feature_importances_'):
            importance = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            importance = np.abs(actual_model.coef_)
        else:
            return None
            
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)
        
    def save_model(self, filepath: str) -> None:
        """Save the trained model and metadata."""
        if not self.is_fitted:
            raise ValueError("No model to save. Please fit the trainer first.")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'feature_engineer': self.feature_engineer,
            'training_results': {
                name: {k: v for k, v in res.items() if k != 'model'}  # Exclude model objects
                for name, res in self.training_results.items()
            }
        }
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        if not self.training_results:
            return {}
            
        summary = {
            'best_model': self.best_model_name,
            'num_features': len(self.feature_names),
            'training_samples': len(self.X_train) if hasattr(self, 'X_train') else None,
            'validation_samples': len(self.X_val) if hasattr(self, 'X_val') else None,
            'test_samples': len(self.X_test) if hasattr(self, 'X_test') else None,
            'model_results': {}
        }
        
        for name, results in self.training_results.items():
            summary['model_results'][name] = {
                'val_r2': results.get('val_r2'),
                'cv_r2_mean': results.get('cv_r2_mean'),
                'cv_r2_std': results.get('cv_r2_std')
            }
            
        return summary


def create_smart_trainer() -> SmartModelTrainer:
    """Create a smart model trainer instance."""
    return SmartModelTrainer()