"""
Base model classes for the roller mill vibration prediction project.
"""

import pickle
import joblib
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


class BaseModel(ABC):
    """Base class for all models in the project."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize the base model.
        
        Args:
            name: Name of the model.
            **kwargs: Additional arguments.
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {}
        self.logger = logging.getLogger(__name__)
        
        # Store initialization parameters
        self.init_params = kwargs
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create the underlying model instance.
        
        Args:
            **kwargs: Model-specific parameters.
            
        Returns:
            Model instance.
        """
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseModel':
        """Fit the model to training data.
        
        Args:
            X: Training features.
            y: Training target.
            **kwargs: Additional fitting parameters.
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Features to predict on.
            
        Returns:
            Predictions array.
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities (if applicable).
        
        Args:
            X: Features to predict on.
            
        Returns:
            Probability predictions or None if not applicable.
        """
        # Default implementation - override if model supports probabilities
        return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores or None.
        """
        # Default implementation - override if model supports feature importance
        return None
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the model on given data.
        
        Args:
            X: Features.
            y: True values.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'r2_score': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }
        
        # Add MAPE if no zero values in y
        if not np.any(y == 0):
            metrics['mape'] = mean_absolute_percentage_error(y, predictions)
        
        return metrics
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'init_params': self.init_params
        }
        
        # Use joblib for sklearn models, pickle for others
        if hasattr(self.model, 'fit') and hasattr(self.model, 'predict'):
            joblib.dump(model_data, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'BaseModel':
        """Load a model from disk.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            Self for method chaining.
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Try joblib first, then pickle
        try:
            model_data = joblib.load(filepath)
        except Exception:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        # Restore model state
        self.model = model_data['model']
        self.name = model_data['name']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data.get('feature_names')
        self.training_history = model_data.get('training_history', {})
        self.init_params = model_data.get('init_params', {})
        
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters.
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return self.init_params.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            Self for method chaining.
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        # Update init_params
        self.init_params.update(params)
        return self
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model."""
        return self.__str__()


class SklearnModel(BaseModel):
    """Wrapper for scikit-learn models."""
    
    def __init__(self, name: str, model_class: type, **kwargs):
        """Initialize the sklearn model wrapper.
        
        Args:
            name: Name of the model.
            model_class: Scikit-learn model class.
            **kwargs: Model parameters.
        """
        super().__init__(name, **kwargs)
        self.model_class = model_class
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs) -> BaseEstimator:
        """Create the sklearn model instance."""
        return self.model_class(**kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SklearnModel':
        """Fit the sklearn model."""
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Fit the model
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        # Store training information
        self.training_history['training_samples'] = len(X)
        self.training_history['features'] = len(X.columns)
        
        self.logger.info(f"Fitted {self.name} with {len(X)} samples and {len(X.columns)} features")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the sklearn model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Ensure feature order matches training
        if self.feature_names is not None:
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities for classification models."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            if self.feature_names is not None:
                X = X[self.feature_names]
            return self.model.predict_proba(X)
        
        return None
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the sklearn model."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        # Try different attribute names for feature importance
        importance_attrs = ['feature_importances_', 'coef_']
        
        for attr in importance_attrs:
            if hasattr(self.model, attr):
                importance_values = getattr(self.model, attr)
                
                # Handle different shapes of importance arrays
                if importance_values.ndim == 1:
                    return dict(zip(self.feature_names, importance_values))
                elif importance_values.ndim == 2 and importance_values.shape[0] == 1:
                    return dict(zip(self.feature_names, importance_values[0]))
        
        return None


class DeepLearningModel(BaseModel):
    """Base class for deep learning models."""
    
    def __init__(self, name: str, **kwargs):
        """Initialize the deep learning model.
        
        Args:
            name: Name of the model.
            **kwargs: Model parameters.
        """
        super().__init__(name, **kwargs)
        self.model = None
        self.scaler = None
        self.history = None
    
    def _prepare_sequences(self, X: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series deep learning models.
        
        Args:
            X: Input features.
            sequence_length: Length of input sequences.
            
        Returns:
            Tuple of (sequences, targets).
        """
        sequences = []
        targets = []
        
        X_values = X.values
        
        for i in range(sequence_length, len(X_values)):
            sequences.append(X_values[i-sequence_length:i])
            targets.append(X_values[i])
        
        return np.array(sequences), np.array(targets)
    
    def _normalize_features(self, X: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Normalize features for deep learning models.
        
        Args:
            X: Input features.
            fit_scaler: Whether to fit the scaler.
            
        Returns:
            Normalized features.
        """
        if fit_scaler or self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    @abstractmethod
    def _build_model(self, input_shape: Tuple[int, ...], **kwargs) -> Any:
        """Build the deep learning model architecture.
        
        Args:
            input_shape: Shape of input data.
            **kwargs: Model parameters.
            
        Returns:
            Compiled model.
        """
        pass
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history metrics.
        
        Returns:
            Dictionary of training metrics over epochs.
        """
        if self.history is not None:
            return self.history.history
        return {}


class EnsembleModel(BaseModel):
    """Ensemble model that combines multiple base models."""
    
    def __init__(self, name: str, base_models: List[BaseModel], method: str = 'average', **kwargs):
        """Initialize the ensemble model.
        
        Args:
            name: Name of the ensemble.
            base_models: List of base models to ensemble.
            method: Ensemble method ('average', 'weighted', 'stacking').
            **kwargs: Additional parameters.
        """
        super().__init__(name, **kwargs)
        self.base_models = base_models
        self.method = method
        self.weights = kwargs.get('weights')
        self.meta_model = kwargs.get('meta_model')
        
        if self.method == 'weighted' and self.weights is None:
            # Equal weights if not specified
            self.weights = [1.0 / len(base_models)] * len(base_models)
    
    def _create_model(self, **kwargs) -> None:
        """Ensemble doesn't have a single underlying model."""
        return None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'EnsembleModel':
        """Fit all base models in the ensemble."""
        self.feature_names = list(X.columns)
        
        # Fit all base models
        for model in self.base_models:
            self.logger.info(f"Fitting base model: {model.name}")
            model.fit(X, y, **kwargs)
        
        # If stacking, fit meta-model
        if self.method == 'stacking' and self.meta_model is not None:
            # Get base model predictions
            base_predictions = np.column_stack([
                model.predict(X) for model in self.base_models
            ])
            
            # Fit meta-model
            self.meta_model.fit(pd.DataFrame(base_predictions), y)
        
        self.is_fitted = True
        self.training_history['base_models'] = len(self.base_models)
        self.training_history['training_samples'] = len(X)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all base models
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        
        if self.method == 'average':
            return np.mean(base_predictions, axis=1)
        
        elif self.method == 'weighted':
            return np.average(base_predictions, weights=self.weights, axis=1)
        
        elif self.method == 'stacking' and self.meta_model is not None:
            return self.meta_model.predict(pd.DataFrame(base_predictions))
        
        else:
            raise ValueError(f"Unsupported ensemble method: {self.method}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get averaged feature importance from base models."""
        if not self.is_fitted or self.feature_names is None:
            return None
        
        importance_dicts = []
        for model in self.base_models:
            importance = model.get_feature_importance()
            if importance is not None:
                importance_dicts.append(importance)
        
        if not importance_dicts:
            return None
        
        # Average importance scores
        avg_importance = {}
        for feature in self.feature_names:
            scores = [imp.get(feature, 0) for imp in importance_dicts]
            avg_importance[feature] = np.mean(scores)
        
        return avg_importance


class ModelFactory:
    """Factory for creating different types of models."""
    
    @staticmethod
    def create_linear_regression(**kwargs) -> SklearnModel:
        """Create a Linear Regression model."""
        from sklearn.linear_model import LinearRegression
        return SklearnModel("LinearRegression", LinearRegression, **kwargs)
    
    @staticmethod
    def create_ridge_regression(**kwargs) -> SklearnModel:
        """Create a Ridge Regression model."""
        from sklearn.linear_model import Ridge
        return SklearnModel("Ridge", Ridge, **kwargs)
    
    @staticmethod
    def create_random_forest(**kwargs) -> SklearnModel:
        """Create a Random Forest model."""
        from sklearn.ensemble import RandomForestRegressor
        return SklearnModel("RandomForest", RandomForestRegressor, **kwargs)
    
    @staticmethod
    def create_xgboost(**kwargs) -> SklearnModel:
        """Create an XGBoost model."""
        import xgboost as xgb
        return SklearnModel("XGBoost", xgb.XGBRegressor, **kwargs)
    
    @staticmethod
    def create_lightgbm(**kwargs) -> SklearnModel:
        """Create a LightGBM model."""
        try:
            import lightgbm as lgb
            return SklearnModel("LightGBM", lgb.LGBMRegressor, **kwargs)
        except ImportError:
            raise ImportError("LightGBM not installed. Install with: pip install lightgbm")
    
    @staticmethod
    def create_catboost(**kwargs) -> SklearnModel:
        """Create a CatBoost model."""
        try:
            from catboost import CatBoostRegressor
            return SklearnModel("CatBoost", CatBoostRegressor, **kwargs)
        except ImportError:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
    
    @staticmethod
    def create_ensemble(base_models: List[BaseModel], method: str = 'average', **kwargs) -> EnsembleModel:
        """Create an ensemble model.
        
        Args:
            base_models: List of base models.
            method: Ensemble method.
            **kwargs: Additional parameters.
            
        Returns:
            Ensemble model.
        """
        return EnsembleModel(f"Ensemble_{method}", base_models, method, **kwargs)


class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self):
        """Initialize the model registry."""
        self.models = {}
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, model: BaseModel) -> None:
        """Register a model in the registry.
        
        Args:
            model: Model to register.
        """
        self.models[model.name] = model
        self.logger.info(f"Registered model: {model.name}")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a model by name.
        
        Args:
            name: Model name.
            
        Returns:
            Model instance or None if not found.
        """
        return self.models.get(name)
    
    def list_models(self) -> List[str]:
        """Get list of registered model names.
        
        Returns:
            List of model names.
        """
        return list(self.models.keys())
    
    def remove_model(self, name: str) -> bool:
        """Remove a model from the registry.
        
        Args:
            name: Model name to remove.
            
        Returns:
            True if model was removed, False if not found.
        """
        if name in self.models:
            del self.models[name]
            self.logger.info(f"Removed model: {name}")
            return True
        return False
    
    def get_best_model(self, metric: str = 'r2_score', X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Optional[BaseModel]:
        """Get the best model based on validation performance.
        
        Args:
            metric: Metric to use for comparison.
            X_val: Validation features.
            y_val: Validation target.
            
        Returns:
            Best model or None if no models available.
        """
        if not self.models or X_val is None or y_val is None:
            return None
        
        best_model = None
        best_score = -np.inf if metric in ['r2_score'] else np.inf
        
        for model in self.models.values():
            if model.is_fitted:
                try:
                    metrics = model.evaluate(X_val, y_val)
                    score = metrics.get(metric)
                    
                    if score is not None:
                        if (metric in ['r2_score'] and score > best_score) or \
                           (metric not in ['r2_score'] and score < best_score):
                            best_score = score
                            best_model = model
                
                except Exception as e:
                    self.logger.warning(f"Error evaluating model {model.name}: {e}")
        
        if best_model:
            self.logger.info(f"Best model: {best_model.name} with {metric}={best_score:.4f}")
        
        return best_model
    
    def save_all_models(self, directory: Union[str, Path]) -> None:
        """Save all registered models to a directory.
        
        Args:
            directory: Directory to save models.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        for model in self.models.values():
            model_path = directory / f"{model.name}.pkl"
            model.save_model(model_path)
        
        self.logger.info(f"Saved {len(self.models)} models to {directory}")
    
    def load_models_from_directory(self, directory: Union[str, Path]) -> int:
        """Load models from a directory.
        
        Args:
            directory: Directory containing saved models.
            
        Returns:
            Number of models loaded.
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.warning(f"Directory not found: {directory}")
            return 0
        
        loaded_count = 0
        
        for model_file in directory.glob("*.pkl"):
            try:
                # Create a dummy model to load the file
                dummy_model = BaseModel("dummy")
                dummy_model.load_model(model_file)
                
                # Register the loaded model
                self.register_model(dummy_model)
                loaded_count += 1
                
            except Exception as e:
                self.logger.error(f"Error loading model from {model_file}: {e}")
        
        self.logger.info(f"Loaded {loaded_count} models from {directory}")
        return loaded_count