"""
Model training utilities for the roller mill vibration prediction project.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error

from config.settings import get_config
from .base_model import BaseModel, ModelFactory, ModelRegistry, SklearnModel, EnsembleModel
from .lstm_model import LSTMModel, GRUModel
from src.evaluation.metrics import ModelEvaluator


class ModelTrainer:
    """Main class for training machine learning models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model trainer.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.registry = ModelRegistry()
        self.evaluator = ModelEvaluator()
        
        # MLflow tracking (if enabled)
        if self.config.mlflow.enabled:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
                mlflow.set_experiment(self.config.mlflow.experiment_name)
                self.mlflow = mlflow
            except ImportError:
                self.logger.warning("MLflow not installed. Model tracking disabled.")
                self.mlflow = None
        else:
            self.mlflow = None
    
    def create_models_from_config(self) -> List[BaseModel]:
        """Create models based on configuration.
        
        Returns:
            List of configured models.
        """
        models = []
        
        # Linear Regression
        if self.config.models.linear_regression.get('enabled', False):
            model = ModelFactory.create_linear_regression()
            models.append(model)
            self.logger.info("Created Linear Regression model")
        
        # Random Forest
        if self.config.models.random_forest.get('enabled', False):
            # Use default parameters for now (hyperparameter tuning will optimize these)
            model = ModelFactory.create_random_forest(
                n_estimators=100,
                random_state=42
            )
            models.append(model)
            self.logger.info("Created Random Forest model")
        
        # XGBoost
        if self.config.models.xgboost.get('enabled', False):
            model = ModelFactory.create_xgboost(
                objective='reg:squarederror',
                random_state=42
            )
            models.append(model)
            self.logger.info("Created XGBoost model")
        
        # LSTM
        if self.config.models.lstm.get('enabled', False):
            lstm_params = self.config.models.lstm.get('params', {})
            model = LSTMModel(
                sequence_length=lstm_params.get('sequence_length', 60),
                units=lstm_params.get('units', [50])[0] if isinstance(lstm_params.get('units', [50]), list) else lstm_params.get('units', 50),
                layers=lstm_params.get('layers', [2])[0] if isinstance(lstm_params.get('layers', [2]), list) else lstm_params.get('layers', 2),
                dropout=lstm_params.get('dropout', [0.2])[0] if isinstance(lstm_params.get('dropout', [0.2]), list) else lstm_params.get('dropout', 0.2),
                learning_rate=lstm_params.get('learning_rate', [0.001])[0] if isinstance(lstm_params.get('learning_rate', [0.001]), list) else lstm_params.get('learning_rate', 0.001),
                batch_size=lstm_params.get('batch_size', [32])[0] if isinstance(lstm_params.get('batch_size', [32]), list) else lstm_params.get('batch_size', 32),
                epochs=lstm_params.get('epochs', 100)
            )
            models.append(model)
            self.logger.info("Created LSTM model")
        
        return models
    
    def train_single_model(
        self, 
        model: BaseModel, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train a single model.
        
        Args:
            model: Model to train.
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            **kwargs: Additional training parameters.
            
        Returns:
            Training results dictionary.
        """
        self.logger.info(f"Training model: {model.name}")
        
        start_time = time.time()
        
        # Start MLflow run if enabled
        if self.mlflow is not None:
            self.mlflow.start_run(run_name=f"{model.name}_training")
            
            # Log model parameters
            params = model.get_params()
            for key, value in params.items():
                if isinstance(value, (int, float, str, bool)):
                    self.mlflow.log_param(key, value)
        
        try:
            # Train the model
            model.fit(X_train, y_train, **kwargs)
            
            training_time = time.time() - start_time
            
            # Evaluate on training data
            train_metrics = model.evaluate(X_train, y_train)
            
            # Evaluate on validation data if provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_metrics = model.evaluate(X_val, y_val)
            
            # Compile results
            results = {
                'model_name': model.name,
                'training_time': training_time,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model': model
            }
            
            # Log metrics to MLflow
            if self.mlflow is not None:
                self.mlflow.log_metric("training_time", training_time)
                
                # Log training metrics
                for metric, value in train_metrics.items():
                    self.mlflow.log_metric(f"train_{metric}", value)
                
                # Log validation metrics
                for metric, value in val_metrics.items():
                    self.mlflow.log_metric(f"val_{metric}", value)
                
                # Log model
                try:
                    self.mlflow.sklearn.log_model(model.model, "model")
                except Exception as e:
                    self.logger.warning(f"Could not log model to MLflow: {e}")
            
            # Register model
            self.registry.register_model(model)
            
            self.logger.info(f"Model {model.name} trained successfully in {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training model {model.name}: {e}")
            return {
                'model_name': model.name,
                'error': str(e)
            }
        
        finally:
            # End MLflow run
            if self.mlflow is not None:
                self.mlflow.end_run()
    
    def hyperparameter_optimization(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 100
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """Perform hyperparameter optimization using Optuna.
        
        Args:
            model_name: Name of the model to optimize.
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            n_trials: Number of optimization trials.
            
        Returns:
            Tuple of (best_model, best_params).
        """
        self.logger.info(f"Starting hyperparameter optimization for {model_name}")
        
        def objective(trial):
            """Optuna objective function."""
            try:
                # Get model-specific parameter suggestions
                if model_name.lower() == 'random_forest' or model_name.lower() == 'randomforest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                        'max_depth': trial.suggest_int('max_depth', 5, 30),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                    }
                    model = ModelFactory.create_random_forest(**params)
                
                elif model_name.lower() == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                    }
                    model = ModelFactory.create_xgboost(
                        objective='reg:squarederror',
                        random_state=42,
                        **params
                    )
                
                elif model_name.lower() == 'lstm':
                    params = {
                        'units': trial.suggest_int('units', 32, 200),
                        'layers': trial.suggest_int('layers', 1, 4),
                        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                    }
                    model = LSTMModel(**params)
                
                else:
                    raise ValueError(f"Hyperparameter optimization not supported for {model_name}")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                predictions = model.predict(X_val)
                
                # Handle NaN predictions (from LSTM sequence requirement)
                valid_mask = ~np.isnan(predictions)
                if valid_mask.sum() == 0:
                    return float('inf')  # Return worst possible score
                
                score = r2_score(y_val[valid_mask], predictions[valid_mask])
                
                return -score  # Optuna minimizes, we want to maximize R²
                
            except Exception as e:
                self.logger.warning(f"Trial failed with error: {e}")
                return float('inf')  # Return worst possible score for failed trials
        
        # Create and run study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_score = -study.best_value  # Convert back to positive R²
        
        self.logger.info(f"Best parameters for {model_name}: {best_params}")
        self.logger.info(f"Best validation R²: {best_score:.4f}")
        
        # Create and train best model
        if model_name.lower() == 'random_forest' or model_name.lower() == 'randomforest':
            best_model = ModelFactory.create_random_forest(**best_params)
        elif model_name.lower() == 'xgboost':
            best_model = ModelFactory.create_xgboost(
                objective='reg:squarederror',
                random_state=42,
                **best_params
            )
        elif model_name.lower() == 'lstm':
            best_model = LSTMModel(**best_params)
        
        # Train the best model on full training data
        best_model.fit(X_train, y_train)
        
        return best_model, best_params
    
    def cross_validate_model(
        self,
        model: BaseModel,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Perform time series cross-validation.
        
        Args:
            model: Model to cross-validate.
            X: Features.
            y: Target.
            cv_folds: Number of CV folds.
            
        Returns:
            Cross-validation results.
        """
        self.logger.info(f"Cross-validating model: {model.name}")
        
        # Use TimeSeriesSplit for proper time series validation
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_scores = []
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"Processing fold {fold + 1}/{cv_folds}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Create a copy of the model for this fold
            fold_model = type(model)(model.name + f"_fold_{fold}", **model.get_params())
            
            # Train model
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            predictions = fold_model.predict(X_val_fold)
            
            # Handle NaN predictions
            valid_mask = ~np.isnan(predictions)
            if valid_mask.sum() > 0:
                fold_score = r2_score(y_val_fold[valid_mask], predictions[valid_mask])
            else:
                fold_score = 0  # Worst possible score
            
            cv_scores.append(fold_score)
            fold_results.append({
                'fold': fold,
                'score': fold_score,
                'train_size': len(X_train_fold),
                'val_size': len(X_val_fold)
            })
        
        # Calculate statistics
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        results = {
            'model_name': model.name,
            'cv_scores': cv_scores,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_results': fold_results
        }
        
        self.logger.info(f"CV Results for {model.name}: {mean_score:.4f} ± {std_score:.4f}")
        
        return results
    
    def train_ensemble(
        self,
        base_models: List[BaseModel],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        methods: List[str] = None
    ) -> List[EnsembleModel]:
        """Train ensemble models.
        
        Args:
            base_models: List of base models for ensemble.
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            methods: Ensemble methods to use.
            
        Returns:
            List of trained ensemble models.
        """
        if methods is None:
            methods = ['average', 'weighted']
        
        ensemble_models = []
        
        # First, train all base models
        trained_base_models = []
        for model in base_models:
            if not model.is_fitted:
                model.fit(X_train, y_train)
            trained_base_models.append(model)
        
        # Create ensemble models
        for method in methods:
            self.logger.info(f"Creating {method} ensemble")
            
            if method == 'weighted' and X_val is not None and y_val is not None:
                # Calculate weights based on validation performance
                weights = []
                for model in trained_base_models:
                    val_metrics = model.evaluate(X_val, y_val)
                    weights.append(val_metrics.get('r2_score', 0))
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                else:
                    weights = [1.0 / len(trained_base_models)] * len(trained_base_models)
                
                ensemble = ModelFactory.create_ensemble(
                    trained_base_models,
                    method=method,
                    weights=weights
                )
            else:
                ensemble = ModelFactory.create_ensemble(trained_base_models, method=method)
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            ensemble_models.append(ensemble)
            
            # Register ensemble
            self.registry.register_model(ensemble)
            
            self.logger.info(f"Created {method} ensemble with {len(trained_base_models)} base models")
        
        return ensemble_models
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimize_hyperparams: bool = True,
        create_ensembles: bool = True
    ) -> Dict[str, Any]:
        """Train all configured models.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features.
            y_val: Validation target.
            optimize_hyperparams: Whether to perform hyperparameter optimization.
            create_ensembles: Whether to create ensemble models.
            
        Returns:
            Training results summary.
        """
        self.logger.info("Starting comprehensive model training")
        
        # Create models from configuration
        base_models = self.create_models_from_config()
        
        training_results = []
        optimized_models = []
        
        # Train each model
        for model in base_models:
            # Basic training
            result = self.train_single_model(model, X_train, y_train, X_val, y_val)
            training_results.append(result)
            
            # Hyperparameter optimization
            if optimize_hyperparams and model.name in ['RandomForest', 'XGBoost', 'LSTM']:
                try:
                    self.logger.info(f"Optimizing hyperparameters for {model.name}")
                    optimized_model, best_params = self.hyperparameter_optimization(
                        model.name,
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        n_trials=self.config.training.optimization.get('n_trials', 50)
                    )
                    
                    # Evaluate optimized model
                    opt_result = self.train_single_model(
                        optimized_model,
                        X_train,
                        y_train,
                        X_val,
                        y_val
                    )
                    opt_result['optimized'] = True
                    opt_result['best_params'] = best_params
                    training_results.append(opt_result)
                    optimized_models.append(optimized_model)
                    
                except Exception as e:
                    self.logger.error(f"Hyperparameter optimization failed for {model.name}: {e}")
                    optimized_models.append(model)  # Use original model
            else:
                optimized_models.append(model)
        
        # Create ensemble models
        ensemble_models = []
        if create_ensembles and len(optimized_models) > 1:
            try:
                ensemble_models = self.train_ensemble(
                    optimized_models,
                    X_train,
                    y_train,
                    X_val,
                    y_val
                )
                
                # Evaluate ensembles
                for ensemble in ensemble_models:
                    ens_result = self.train_single_model(ensemble, X_train, y_train, X_val, y_val)
                    ens_result['ensemble'] = True
                    training_results.append(ens_result)
                    
            except Exception as e:
                self.logger.error(f"Ensemble training failed: {e}")
        
        # Compile final results
        results_summary = {
            'total_models_trained': len(training_results),
            'base_models': len(base_models),
            'optimized_models': len(optimized_models),
            'ensemble_models': len(ensemble_models),
            'training_results': training_results,
            'best_model': self.registry.get_best_model('r2_score', X_val, y_val)
        }
        
        self.logger.info(f"Training completed. {len(training_results)} models trained.")
        
        if results_summary['best_model']:
            best_metrics = results_summary['best_model'].evaluate(X_val, y_val)
            self.logger.info(f"Best model: {results_summary['best_model'].name} (R²: {best_metrics['r2_score']:.4f})")
        
        return results_summary
    
    def save_training_results(self, results: Dict[str, Any], filepath: Union[str, Path]) -> None:
        """Save training results to file.
        
        Args:
            results: Training results dictionary.
            filepath: Path to save results.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save models
        model_dir = filepath.parent / "models"
        self.registry.save_all_models(model_dir)
        
        # Save results summary (without model objects)
        import json
        summary_data = {
            'total_models_trained': results['total_models_trained'],
            'base_models': results['base_models'],
            'optimized_models': results['optimized_models'],
            'ensemble_models': results['ensemble_models'],
            'best_model_name': results['best_model'].name if results['best_model'] else None
        }
        
        # Add training results (excluding model objects)
        summary_data['model_results'] = []
        for result in results['training_results']:
            clean_result = {k: v for k, v in result.items() if k != 'model'}
            summary_data['model_results'].append(clean_result)
        
        with open(filepath, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        self.logger.info(f"Training results saved to {filepath}")


def train_models_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config_path: Optional[str] = None,
    save_results: bool = True
) -> Dict[str, Any]:
    """Complete model training pipeline.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        config_path: Path to configuration file.
        save_results: Whether to save results.
        
    Returns:
        Training results dictionary.
    """
    # Initialize trainer
    trainer = ModelTrainer(config_path)
    
    # Train all models
    results = trainer.train_all_models(
        X_train, y_train, X_val, y_val,
        optimize_hyperparams=True,
        create_ensembles=True
    )
    
    # Save results if requested
    if save_results:
        config = get_config()
        results_path = Path(config.paths.experiments) / "training_results.json"
        trainer.save_training_results(results, results_path)
    
    return results