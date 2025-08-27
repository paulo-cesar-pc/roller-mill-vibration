"""
Ensemble trainer specifically designed for noisy industrial time series.
Combines multiple robust approaches for improved prediction on noisy vibration data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, TheilSenRegressor, RANSACRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin, clone
import joblib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NoisyEnsembleConfig:
    """Configuration for noisy data ensemble models."""
    # Model weights and selection
    use_robust_regressors: bool = True
    use_ensemble_methods: bool = True
    use_quantile_regression: bool = True
    use_denoising_approach: bool = False  # Requires additional setup
    
    # Cross-validation
    cv_folds: int = 5
    cv_method: str = 'time_series'  # 'time_series' or 'stratified'
    
    # Robust model parameters
    huber_epsilon: float = 1.35
    ransac_residual_threshold: float = None
    theil_sen_max_subpopulation: int = 1000
    
    # Ensemble parameters
    rf_n_estimators: int = 200
    gb_n_estimators: int = 150
    gb_learning_rate: float = 0.1
    
    # Model selection criteria
    primary_metric: str = 'r2'  # 'r2', 'mae', 'mse'
    robustness_weight: float = 0.3  # Weight for robustness vs accuracy
    
    # Quantile regression settings
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])


class QuantileRegressor(BaseEstimator, RegressorMixin):
    """Wrapper for quantile regression using GradientBoostingRegressor."""
    
    def __init__(self, quantile=0.5, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.quantile = quantile
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = None
        
    def fit(self, X, y):
        """Fit quantile regression model."""
        self.model = GradientBoostingRegressor(
            loss='quantile',
            alpha=self.quantile,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict using fitted quantile model."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class RobustEnsemble(BaseEstimator, RegressorMixin):
    """Meta-estimator that combines multiple robust regressors."""
    
    def __init__(self, base_estimators, weights=None, use_stacking=False):
        self.base_estimators = base_estimators
        self.weights = weights
        self.use_stacking = use_stacking
        self.fitted_estimators = []
        self.meta_model = None
        
    def fit(self, X, y):
        """Fit all base estimators."""
        self.fitted_estimators = []
        
        for estimator in self.base_estimators:
            fitted_est = clone(estimator)
            fitted_est.fit(X, y)
            self.fitted_estimators.append(fitted_est)
            
        # If stacking, fit meta-model
        if self.use_stacking:
            # Create meta-features
            meta_features = np.column_stack([
                est.predict(X) for est in self.fitted_estimators
            ])
            
            # Use robust meta-learner
            self.meta_model = HuberRegressor()
            self.meta_model.fit(meta_features, y)
            
        return self
    
    def predict(self, X):
        """Predict using ensemble of fitted estimators."""
        predictions = np.column_stack([
            est.predict(X) for est in self.fitted_estimators
        ])
        
        if self.use_stacking and self.meta_model is not None:
            return self.meta_model.predict(predictions)
        else:
            # Weighted average
            if self.weights is not None:
                return np.average(predictions, axis=1, weights=self.weights)
            else:
                return np.mean(predictions, axis=1)


class NoisyEnsembleTrainer:
    """Trainer for ensembles specifically designed for noisy industrial data."""
    
    def __init__(self, config: Optional[NoisyEnsembleConfig] = None):
        """Initialize with configuration."""
        self.config = config or NoisyEnsembleConfig()
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.cv_results = {}
        self.fitted = False
        
    def _create_robust_regressors(self) -> Dict[str, BaseEstimator]:
        """Create robust regression models for noisy data."""
        models = {}
        
        # Huber Regressor - robust to outliers
        models['huber'] = HuberRegressor(
            epsilon=self.config.huber_epsilon,
            max_iter=200,
            alpha=0.0001
        )
        
        # RANSAC - very robust to outliers
        models['ransac'] = RANSACRegressor(
            estimator=None,  # Uses LinearRegression by default
            residual_threshold=self.config.ransac_residual_threshold,
            max_trials=100,
            random_state=42
        )
        
        # Theil-Sen - robust to outliers, good for small datasets
        models['theil_sen'] = TheilSenRegressor(
            max_subpopulation=min(self.config.theil_sen_max_subpopulation, 1000),
            random_state=42
        )
        
        # SVR with RBF kernel - nonlinear, robust with proper parameters
        models['svr_rbf'] = SVR(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            epsilon=0.1
        )
        
        # SVR with linear kernel - simpler, more robust
        models['svr_linear'] = SVR(
            kernel='linear',
            C=1.0,
            epsilon=0.1
        )
        
        return models
    
    def _create_ensemble_models(self) -> Dict[str, BaseEstimator]:
        """Create ensemble models for improved robustness."""
        models = {}
        
        # Random Forest - naturally robust to outliers
        models['random_forest'] = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_features='sqrt',
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - with robust loss
        models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=self.config.gb_n_estimators,
            learning_rate=self.config.gb_learning_rate,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            loss='huber',  # Robust loss function
            alpha=0.9,
            random_state=42
        )
        
        # Extra Trees - more random, often more robust
        from sklearn.ensemble import ExtraTreesRegressor
        models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_features='sqrt',
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        return models
    
    def _create_quantile_models(self) -> Dict[str, BaseEstimator]:
        """Create quantile regression models."""
        models = {}
        
        for q in self.config.quantiles:
            models[f'quantile_{int(q*100)}'] = QuantileRegressor(
                quantile=q,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4
            )
            
        return models
    
    def _evaluate_model_robustness(
        self, 
        model: BaseEstimator, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model robustness using cross-validation with noise injection."""
        if self.config.cv_method == 'time_series':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            
        # Original performance
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        
        # Performance with noise injection
        noise_level = y.std() * 0.1  # 10% of target std as noise
        y_noisy = y + np.random.normal(0, noise_level, size=len(y))
        
        r2_scores_noisy = cross_val_score(model, X, y_noisy, cv=cv, scoring='r2')
        
        # Robustness metrics
        robustness_score = 1 - abs(r2_scores.mean() - r2_scores_noisy.mean()) / (abs(r2_scores.mean()) + 1e-6)
        
        return {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_mean': -mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'robustness': robustness_score
        }
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Fit ensemble of robust models."""
        self.logger.info("Training ensemble of robust models for noisy data")
        
        # Create all model types
        all_models = {}
        
        if self.config.use_robust_regressors:
            all_models.update(self._create_robust_regressors())
            
        if self.config.use_ensemble_methods:
            all_models.update(self._create_ensemble_models())
            
        if self.config.use_quantile_regression:
            all_models.update(self._create_quantile_models())
        
        # Prepare data scaling
        self.scalers['robust'] = RobustScaler()
        self.scalers['standard'] = StandardScaler()
        
        X_robust = pd.DataFrame(
            self.scalers['robust'].fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        X_standard = pd.DataFrame(
            self.scalers['standard'].fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Train and evaluate all models
        training_results = {}
        
        for name, model in all_models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Choose appropriate scaling
                if name in ['svr_rbf', 'svr_linear']:
                    X_scaled = X_standard
                elif name.startswith('quantile'):
                    X_scaled = X_robust
                else:
                    X_scaled = X_robust if 'robust' in name or name in ['huber', 'ransac', 'theil_sen'] else X
                
                # Fit model
                if sample_weight is not None and hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                    model.fit(X_scaled, y, sample_weight=sample_weight)
                else:
                    model.fit(X_scaled, y)
                
                # Evaluate robustness
                eval_results = self._evaluate_model_robustness(model, X_scaled, y)
                
                # Store results
                training_results[name] = eval_results
                self.models[name] = model
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
                elif hasattr(model, 'coef_'):
                    self.feature_importance[name] = dict(zip(X.columns, np.abs(model.coef_)))
                    
                self.logger.info(f"{name} - R²: {eval_results['r2_mean']:.4f}±{eval_results['r2_std']:.4f}, "
                               f"Robustness: {eval_results['robustness']:.4f}")
                               
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                training_results[name] = {'error': str(e)}
        
        # Create meta-ensemble from best models
        self._create_meta_ensemble(X_robust, y, training_results)
        
        self.cv_results = training_results
        self.fitted = True
        
        self.logger.info(f"Successfully trained {len(self.models)} robust models")
        return training_results
    
    def _create_meta_ensemble(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        training_results: Dict[str, Dict]
    ):
        """Create meta-ensemble from best performing models."""
        # Select best models based on combined score
        model_scores = {}
        for name, results in training_results.items():
            if 'error' not in results:
                # Combined score: accuracy + robustness
                accuracy_score = results.get('r2_mean', 0)
                robustness_score = results.get('robustness', 0)
                combined_score = (1 - self.config.robustness_weight) * accuracy_score + \
                               self.config.robustness_weight * robustness_score
                model_scores[name] = combined_score
        
        # Select top models
        if model_scores:
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            top_models = sorted_models[:5]  # Top 5 models
            
            selected_estimators = [self.models[name] for name, _ in top_models]
            weights = np.array([score for _, score in top_models])
            weights = weights / weights.sum()  # Normalize
            
            # Create meta-ensemble
            self.models['meta_ensemble'] = RobustEnsemble(
                base_estimators=selected_estimators,
                weights=weights,
                use_stacking=True
            )
            
            # Fit meta-ensemble
            try:
                self.models['meta_ensemble'].fit(X, y)
                self.logger.info(f"Created meta-ensemble from {len(selected_estimators)} models")
            except Exception as e:
                self.logger.error(f"Failed to create meta-ensemble: {e}")
    
    def predict(
        self, 
        X: pd.DataFrame, 
        model_name: Optional[str] = None,
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with specified or best model."""
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")
            
        if model_name is None:
            # Use meta-ensemble if available, otherwise best single model
            if 'meta_ensemble' in self.models:
                model_name = 'meta_ensemble'
            else:
                # Select best model from CV results
                best_model = max(
                    [(name, results.get('r2_mean', -np.inf)) for name, results in self.cv_results.items() 
                     if 'error' not in results],
                    key=lambda x: x[1]
                )[0]
                model_name = best_model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model = self.models[model_name]
        
        # Apply appropriate scaling
        if model_name in ['svr_rbf', 'svr_linear']:
            X_scaled = pd.DataFrame(
                self.scalers['standard'].transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scalers['robust'].transform(X),
                columns=X.columns,
                index=X.index
            )
        
        predictions = model.predict(X_scaled)
        
        if return_uncertainty:
            # Estimate uncertainty using ensemble of quantile models
            uncertainty = self._estimate_uncertainty(X_scaled)
            return predictions, uncertainty
        
        return predictions
    
    def _estimate_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate prediction uncertainty using quantile models."""
        quantile_predictions = {}
        
        for name, model in self.models.items():
            if name.startswith('quantile_'):
                try:
                    pred = model.predict(X)
                    quantile_value = float(name.split('_')[1]) / 100
                    quantile_predictions[quantile_value] = pred
                except Exception:
                    continue
        
        if len(quantile_predictions) >= 2:
            # Use IQR as uncertainty measure
            q25 = quantile_predictions.get(0.25, quantile_predictions.get(0.1))
            q75 = quantile_predictions.get(0.75, quantile_predictions.get(0.9))
            
            if q25 is not None and q75 is not None:
                return q75 - q25
                
        # Fallback: use ensemble variance
        if len(self.models) > 1:
            predictions = []
            for name, model in self.models.items():
                if not name.startswith('quantile_') and name != 'meta_ensemble':
                    try:
                        pred = model.predict(X)
                        predictions.append(pred)
                    except Exception:
                        continue
            
            if predictions:
                return np.std(predictions, axis=0)
        
        return np.zeros(len(X))
    
    def get_best_model_name(self) -> str:
        """Get name of best performing model."""
        if 'meta_ensemble' in self.models:
            return 'meta_ensemble'
        
        best_model = max(
            [(name, results.get('r2_mean', -np.inf)) for name, results in self.cv_results.items() 
             if 'error' not in results],
            key=lambda x: x[1]
        )[0]
        return best_model
    
    def save_models(self, filepath: str):
        """Save trained models and scalers."""
        save_data = {
            'models': self.models,
            'scalers': self.scalers,
            'cv_results': self.cv_results,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        joblib.dump(save_data, filepath)
        self.logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models and scalers."""
        save_data = joblib.load(filepath)
        self.models = save_data['models']
        self.scalers = save_data['scalers']
        self.cv_results = save_data['cv_results']
        self.feature_importance = save_data.get('feature_importance', {})
        self.config = save_data.get('config', self.config)
        self.fitted = True
        self.logger.info(f"Models loaded from {filepath}")


def create_noisy_ensemble_config(
    focus_on_robustness: bool = True,
    include_quantiles: bool = True
) -> NoisyEnsembleConfig:
    """Create configuration for noisy data ensemble."""
    return NoisyEnsembleConfig(
        use_robust_regressors=True,
        use_ensemble_methods=True,
        use_quantile_regression=include_quantiles,
        robustness_weight=0.4 if focus_on_robustness else 0.2,
        huber_epsilon=1.35 if focus_on_robustness else 1.5,
        cv_folds=5,
        primary_metric='r2'
    )