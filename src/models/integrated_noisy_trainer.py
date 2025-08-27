"""
Integrated training pipeline for noisy industrial vibration data.
Combines noise reduction, specialized feature engineering, robust modeling, and alternative formulations.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import joblib
import json

# Import our specialized modules
from src.data.noise_reduction import IndustrialNoiseReducer, NoiseReductionConfig
from src.features.mill_feature_engineer import MillFeatureEngineer, MillFeatureConfig
from src.models.noisy_ensemble_trainer import NoisyEnsembleTrainer, NoisyEnsembleConfig
from src.models.alternative_formulations import AlternativeFormulations, AlternativeFormulationConfig
from src.evaluation.time_series_evaluator import TimeSeriesEvaluator


@dataclass
class IntegratedNoisyConfig:
    """Unified configuration for the integrated noisy data pipeline."""
    # Data processing
    target_frequency: str = '5min'  # Aggregation frequency
    testing_mode: bool = False
    
    # Noise reduction settings
    aggressive_filtering: bool = False
    outlier_detection_method: str = 'iqr'  # 'iqr', 'hampel', 'zscore'
    
    # Feature engineering
    create_mill_features: bool = True
    create_multi_scale_features: bool = True
    feature_selection_method: str = 'importance'  # 'importance', 'correlation', 'none'
    max_features: Optional[int] = 200  # Limit number of features
    
    # Model training
    focus_on_robustness: bool = True
    use_ensemble_methods: bool = True
    use_quantile_regression: bool = True
    
    # Alternative formulations
    use_classification: bool = True
    use_anomaly_detection: bool = True
    use_threshold_monitoring: bool = True
    vibration_thresholds: List[float] = field(default_factory=lambda: [4.5, 6.5, 8.5])
    
    # Validation
    validation_method: str = 'time_series'  # 'time_series', 'forward_chaining'
    n_validation_folds: int = 5
    
    # Output
    save_intermediate_results: bool = True
    create_evaluation_plots: bool = True
    output_dir: str = 'outputs'


class IntegratedNoisyTrainer:
    """
    Integrated trainer for noisy industrial vibration data.
    Provides comprehensive approach with noise reduction, specialized modeling, and alternative formulations.
    """
    
    def __init__(self, config: Optional[IntegratedNoisyConfig] = None):
        """Initialize the integrated trainer."""
        self.config = config or IntegratedNoisyConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.noise_reducer = None
        self.feature_engineer = None
        self.ensemble_trainer = None
        self.alternative_formulations = None
        self.evaluator = TimeSeriesEvaluator()
        
        # Results storage
        self.training_results = {}
        self.feature_names = []
        self.fitted = False
        
        # Create output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
    def _setup_components(self):
        """Initialize all pipeline components."""
        # Noise reduction
        noise_config = NoiseReductionConfig(
            target_frequency=self.config.target_frequency,
            savgol_window=21 if self.config.aggressive_filtering else 11,
            outlier_method=self.config.outlier_detection_method,
            iqr_multiplier=2.0 if self.config.aggressive_filtering else 2.5
        )
        self.noise_reducer = IndustrialNoiseReducer(noise_config)
        
        # Feature engineering
        if self.config.create_mill_features:
            mill_config = MillFeatureConfig(
                create_efficiency_features=True,
                create_stability_features=True,
                create_interaction_features=self.config.create_multi_scale_features,
                create_change_features=self.config.create_multi_scale_features
            )
            self.feature_engineer = MillFeatureEngineer(mill_config)
        
        # Ensemble trainer
        ensemble_config = NoisyEnsembleConfig(
            use_robust_regressors=True,
            use_ensemble_methods=self.config.use_ensemble_methods,
            use_quantile_regression=self.config.use_quantile_regression,
            robustness_weight=0.4 if self.config.focus_on_robustness else 0.2,
            cv_folds=self.config.n_validation_folds
        )
        self.ensemble_trainer = NoisyEnsembleTrainer(ensemble_config)
        
        # Alternative formulations
        if any([self.config.use_classification, self.config.use_anomaly_detection, self.config.use_threshold_monitoring]):
            alt_config = AlternativeFormulationConfig(
                vibration_bins=self.config.vibration_thresholds,
                balance_classes=True,
                isolation_forest_contamination=0.1
            )
            self.alternative_formulations = AlternativeFormulations(alt_config)
    
    def preprocess_data(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply comprehensive data preprocessing."""
        self.logger.info("Starting comprehensive data preprocessing")
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
        
        # Step 1: Noise reduction and temporal aggregation
        self.logger.info("Applying noise reduction and temporal aggregation")
        df_processed = self.noise_reducer.fit_transform(df, target_col, feature_cols)
        
        # Extract target columns (multiple aggregation statistics)
        target_cols = [col for col in df_processed.columns if col.startswith(target_col)]
        self.logger.info(f"Target columns after aggregation: {target_cols}")
        
        # Use primary target (mean aggregation)
        primary_target_col = f"{target_col}_mean"
        if primary_target_col not in df_processed.columns:
            # Fallback to first available target column
            primary_target_col = target_cols[0] if target_cols else target_col
            
        y_processed = df_processed[primary_target_col].copy()
        X_processed = df_processed.drop(columns=target_cols)
        
        self.logger.info(f"After noise reduction: {X_processed.shape[1]} features, {len(y_processed)} samples")
        
        # Step 2: Specialized feature engineering
        if self.feature_engineer is not None:
            self.logger.info("Applying mill-specific feature engineering")
            combined_df = X_processed.copy()
            combined_df[primary_target_col] = y_processed
            
            df_engineered = self.feature_engineer.fit_transform(combined_df, primary_target_col)
            
            # Separate features and target again
            y_processed = df_engineered[primary_target_col].copy()
            X_processed = df_engineered.drop(columns=[primary_target_col])
            
            self.logger.info(f"After feature engineering: {X_processed.shape[1]} features")
        
        # Step 3: Feature selection
        if self.config.feature_selection_method != 'none' and self.config.max_features:
            X_processed = self._apply_feature_selection(X_processed, y_processed)
        
        # Step 4: Final cleaning
        X_processed, y_processed = self._clean_final_data(X_processed, y_processed)
        
        self.feature_names = list(X_processed.columns)
        self.logger.info(f"Final preprocessed data: {X_processed.shape[1]} features, {len(y_processed)} samples")
        
        return X_processed, y_processed
    
    def _apply_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        self.logger.info(f"Applying feature selection (method: {self.config.feature_selection_method})")
        
        if self.config.feature_selection_method == 'importance':
            # Use RandomForest feature importance
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import SelectFromModel
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            selector = SelectFromModel(rf, max_features=self.config.max_features)
            X_selected = selector.fit_transform(X, y)
            
            selected_features = X.columns[selector.get_support()].tolist()
            X_reduced = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        elif self.config.feature_selection_method == 'correlation':
            # Remove highly correlated features
            correlation_matrix = X.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            X_reduced = X.drop(columns=to_drop)
            
            # Limit to max_features if still too many
            if len(X_reduced.columns) > self.config.max_features:
                # Select by target correlation
                target_corr = abs(X_reduced.corrwith(y))
                top_features = target_corr.nlargest(self.config.max_features).index
                X_reduced = X_reduced[top_features]
        
        else:
            X_reduced = X
        
        self.logger.info(f"Feature selection: {X.shape[1]} -> {X_reduced.shape[1]} features")
        return X_reduced
    
    def _clean_final_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Final data cleaning and validation."""
        # Remove any remaining NaN values
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[valid_mask].copy()
        y_clean = y[valid_mask].copy()
        
        # Remove infinite values
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        X_clean = X_clean.fillna(X_clean.median())
        
        # Remove constant features
        constant_features = X_clean.columns[X_clean.nunique() <= 1].tolist()
        if constant_features:
            self.logger.info(f"Removing {len(constant_features)} constant features")
            X_clean = X_clean.drop(columns=constant_features)
        
        self.logger.info(f"Data cleaning: {len(X)} -> {len(X_clean)} samples")
        return X_clean, y_clean
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target_col: str,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fit the complete integrated pipeline."""
        self.logger.info("Starting integrated noisy data training pipeline")
        
        start_time = datetime.now()
        
        # Setup components
        self._setup_components()
        
        # Preprocess data
        X_processed, y_processed = self.preprocess_data(df, target_col, feature_cols)
        
        results = {
            'preprocessing': {
                'original_shape': df.shape,
                'processed_shape': (len(X_processed), len(X_processed.columns)),
                'feature_names': self.feature_names,
                'target_stats': {
                    'mean': float(y_processed.mean()),
                    'std': float(y_processed.std()),
                    'min': float(y_processed.min()),
                    'max': float(y_processed.max()),
                    'median': float(y_processed.median())
                }
            }
        }
        
        # Train ensemble models
        self.logger.info("Training robust ensemble models")
        ensemble_results = self.ensemble_trainer.fit(X_processed, y_processed)
        results['ensemble_models'] = ensemble_results
        
        # Train alternative formulations
        if self.alternative_formulations is not None:
            self.logger.info("Training alternative formulations")
            alt_results = self.alternative_formulations.fit(X_processed, y_processed)
            results['alternative_formulations'] = alt_results
        
        # Model evaluation
        results['evaluation'] = self._comprehensive_evaluation(X_processed, y_processed)
        
        # Save intermediate results
        if self.config.save_intermediate_results:
            self._save_intermediate_results(results)
        
        self.training_results = results
        self.fitted = True
        
        training_time = datetime.now() - start_time
        self.logger.info(f"Integrated training completed in {training_time}")
        
        return results
    
    def _comprehensive_evaluation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive evaluation of all models."""
        self.logger.info("Performing comprehensive model evaluation")
        
        evaluation_results = {}
        
        # Time series split for evaluation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Evaluate ensemble models
        if self.ensemble_trainer.fitted:
            best_model_name = self.ensemble_trainer.get_best_model_name()
            self.logger.info(f"Evaluating best ensemble model: {best_model_name}")
            
            # Cross-validation evaluation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                
                # Fit and predict
                temp_trainer = NoisyEnsembleTrainer(self.ensemble_trainer.config)
                temp_trainer.scalers = self.ensemble_trainer.scalers  # Reuse scalers
                temp_trainer.fit(X_train_cv, y_train_cv)
                
                y_pred_cv = temp_trainer.predict(X_val_cv, best_model_name)
                
                # Calculate metrics
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                cv_score = {
                    'r2': r2_score(y_val_cv, y_pred_cv),
                    'rmse': np.sqrt(mean_squared_error(y_val_cv, y_pred_cv)),
                    'mae': mean_absolute_error(y_val_cv, y_pred_cv)
                }
                cv_scores.append(cv_score)
            
            # Average CV scores
            evaluation_results['ensemble_cv'] = {
                'r2_mean': np.mean([s['r2'] for s in cv_scores]),
                'r2_std': np.std([s['r2'] for s in cv_scores]),
                'rmse_mean': np.mean([s['rmse'] for s in cv_scores]),
                'mae_mean': np.mean([s['mae'] for s in cv_scores]),
            }
            
            # Full dataset predictions for detailed analysis
            y_pred_full = self.ensemble_trainer.predict(X, best_model_name, return_uncertainty=True)
            if isinstance(y_pred_full, tuple):
                y_pred, uncertainty = y_pred_full
            else:
                y_pred, uncertainty = y_pred_full, None
            
            # Comprehensive evaluation using TimeSeriesEvaluator
            eval_report = self.evaluator.create_evaluation_report(
                y_true=y,
                y_pred=y_pred,
                timestamps=X.index,
                model_name=best_model_name
            )
            
            evaluation_results['detailed_metrics'] = eval_report
            
            # Create evaluation plots if requested
            if self.config.create_evaluation_plots:
                fig = self.evaluator.create_evaluation_plots(
                    y_true=y,
                    y_pred=y_pred,
                    timestamps=X.index,
                    model_name=best_model_name,
                    figsize=(20, 15)
                )
                
                plot_path = self.output_dir / 'plots' / f'evaluation_{best_model_name}.png'
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                evaluation_results['plot_path'] = str(plot_path)
        
        return evaluation_results
    
    def _save_intermediate_results(self, results: Dict[str, Any]):
        """Save intermediate results and models."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save training results
        results_path = self.output_dir / 'reports' / f'training_results_{timestamp}.json'
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        results_json = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=str)
        
        self.logger.info(f"Training results saved to {results_path}")
        
        # Save models
        models_path = self.output_dir / 'models' / f'integrated_models_{timestamp}.pkl'
        model_data = {
            'ensemble_trainer': self.ensemble_trainer,
            'alternative_formulations': self.alternative_formulations,
            'noise_reducer': self.noise_reducer,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, models_path)
        self.logger.info(f"Models saved to {models_path}")
    
    def predict(
        self, 
        X: pd.DataFrame,
        return_all_formulations: bool = False
    ) -> Union[pd.Series, Dict[str, Any]]:
        """Make predictions using the trained pipeline."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Apply same preprocessing
        if self.noise_reducer is not None:
            # Note: This is simplified - in practice, you'd need to handle the target column properly
            X_processed = self.noise_reducer.transform(X, 'dummy_target')
        else:
            X_processed = X
        
        if self.feature_engineer is not None:
            X_processed = self.feature_engineer.transform(X_processed)
        
        # Ensure we have the same features as training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X_processed.columns)
            if missing_features:
                self.logger.warning(f"Missing features for prediction: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X_processed[feature] = 0
            
            # Select only training features in the same order
            X_processed = X_processed[self.feature_names]
        
        if return_all_formulations and self.alternative_formulations is not None:
            # Return comprehensive predictions
            results = {}
            
            # Ensemble prediction
            if self.ensemble_trainer.fitted:
                ensemble_pred = self.ensemble_trainer.predict(X_processed, return_uncertainty=True)
                if isinstance(ensemble_pred, tuple):
                    results['predictions'] = pd.Series(ensemble_pred[0], index=X.index)
                    results['uncertainty'] = pd.Series(ensemble_pred[1], index=X.index)
                else:
                    results['predictions'] = pd.Series(ensemble_pred, index=X.index)
            
            # Alternative formulations
            alt_results = self.alternative_formulations.predict_all(X_processed)
            results.update(alt_results)
            
            return results
        
        else:
            # Return simple prediction from best model
            if self.ensemble_trainer.fitted:
                predictions = self.ensemble_trainer.predict(X_processed)
                return pd.Series(predictions, index=X.index)
            else:
                raise ValueError("No trained models available for prediction")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from all models."""
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before getting feature importance")
        
        importance_data = []
        
        if self.ensemble_trainer.fitted:
            for model_name, importance_dict in self.ensemble_trainer.feature_importance.items():
                for feature, importance in importance_dict.items():
                    importance_data.append({
                        'model': model_name,
                        'feature': feature,
                        'importance': importance
                    })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            # Aggregate by feature
            feature_importance = importance_df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
            feature_importance = feature_importance.sort_values('mean', ascending=False)
            return feature_importance
        
        return pd.DataFrame()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        if not self.fitted:
            return {}
        
        summary = {
            'pipeline_config': {
                'target_frequency': self.config.target_frequency,
                'aggressive_filtering': self.config.aggressive_filtering,
                'create_mill_features': self.config.create_mill_features,
                'max_features': self.config.max_features,
                'focus_on_robustness': self.config.focus_on_robustness
            },
            'data_summary': self.training_results.get('preprocessing', {}),
            'model_performance': {},
            'feature_count': len(self.feature_names),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Add best model performance
        if 'evaluation' in self.training_results and 'ensemble_cv' in self.training_results['evaluation']:
            cv_results = self.training_results['evaluation']['ensemble_cv']
            summary['model_performance'] = {
                'cv_r2_mean': cv_results.get('r2_mean', 0),
                'cv_r2_std': cv_results.get('r2_std', 0),
                'cv_rmse_mean': cv_results.get('rmse_mean', 0),
                'cv_mae_mean': cv_results.get('mae_mean', 0)
            }
        
        return summary


def create_integrated_config(
    target_frequency: str = '5min',
    focus_on_robustness: bool = True,
    include_all_formulations: bool = True
) -> IntegratedNoisyConfig:
    """Create integrated configuration for different use cases."""
    return IntegratedNoisyConfig(
        target_frequency=target_frequency,
        aggressive_filtering=focus_on_robustness,
        create_mill_features=True,
        create_multi_scale_features=True,
        max_features=200 if focus_on_robustness else 300,
        focus_on_robustness=focus_on_robustness,
        use_ensemble_methods=True,
        use_quantile_regression=True,
        use_classification=include_all_formulations,
        use_anomaly_detection=include_all_formulations,
        use_threshold_monitoring=include_all_formulations,
        save_intermediate_results=True,
        create_evaluation_plots=True
    )