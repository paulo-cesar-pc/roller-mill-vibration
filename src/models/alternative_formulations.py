"""
Alternative problem formulations for noisy vibration data.
Includes classification, anomaly detection, and threshold monitoring approaches.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AlternativeFormulationConfig:
    """Configuration for alternative problem formulations."""
    # Classification settings
    vibration_bins: List[float] = field(default_factory=lambda: [4.0, 6.0, 8.0])  # Threshold boundaries
    classification_labels: List[str] = field(default_factory=lambda: ['Low', 'Normal', 'High', 'Critical'])
    balance_classes: bool = True
    
    # Anomaly detection settings
    isolation_forest_contamination: float = 0.1
    one_class_svm_nu: float = 0.1
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 10
    
    # Threshold monitoring
    operational_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'warning': 6.5,
        'alarm': 8.0,
        'emergency': 9.5
    })
    
    # Change detection
    change_detection_window: int = 20
    change_threshold_std: float = 2.0
    
    # Model evaluation
    cv_folds: int = 5
    test_size: float = 0.2


class VibrationClassifier:
    """Transform vibration prediction to classification problem."""
    
    def __init__(self, config: AlternativeFormulationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self.fitted = False
        
    def _create_vibration_classes(self, y: pd.Series) -> pd.Series:
        """Convert continuous vibration to discrete classes."""
        bins = [-np.inf] + self.config.vibration_bins + [np.inf]
        labels = self.config.classification_labels
        
        y_classes = pd.cut(y, bins=bins, labels=labels, include_lowest=True)
        
        self.logger.info(f"Class distribution:")
        for label, count in y_classes.value_counts().items():
            percentage = count / len(y_classes) * 100
            self.logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        
        return y_classes
    
    def _balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balance classes using SMOTE or undersampling."""
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.combine import SMOTEENN
            
            # Use SMOTEENN for balanced sampling
            smote_enn = SMOTEENN(random_state=42)
            X_balanced, y_balanced = smote_enn.fit_resample(X, y)
            
            self.logger.info(f"Balanced dataset from {len(X)} to {len(X_balanced)} samples")
            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)
            
        except ImportError:
            self.logger.warning("imbalanced-learn not available. Skipping class balancing.")
            return X, y
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit classification models."""
        self.logger.info("Training vibration classification models")
        
        # Convert to classes
        y_classes = self._create_vibration_classes(y)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_classes.astype(str))
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Balance classes if requested
        if self.config.balance_classes:
            X_scaled, y_encoded = self._balance_classes(X_scaled, pd.Series(y_encoded))
        
        # Define classifiers
        classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='ovr'
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        }
        
        # Train and evaluate models
        results = {}
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        for name, classifier in classifiers.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(classifier, X_scaled, y_encoded, cv=cv, scoring='accuracy')
                
                # Fit full model
                classifier.fit(X_scaled, y_encoded)
                self.models[name] = classifier
                
                # Feature importance
                if hasattr(classifier, 'feature_importances_'):
                    feature_importance = dict(zip(X.columns, classifier.feature_importances_))
                elif hasattr(classifier, 'coef_'):
                    feature_importance = dict(zip(X.columns, np.abs(classifier.coef_).mean(axis=0)))
                else:
                    feature_importance = {}
                
                results[name] = {
                    'cv_accuracy_mean': cv_scores.mean(),
                    'cv_accuracy_std': cv_scores.std(),
                    'feature_importance': feature_importance
                }
                
                self.logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.fitted = True
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str = 'random_forest') -> pd.Series:
        """Predict vibration classes."""
        if not self.fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        y_pred_encoded = self.models[model_name].predict(X_scaled)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return pd.Series(y_pred_labels, index=X.index)
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = 'random_forest') -> pd.DataFrame:
        """Predict class probabilities."""
        if not self.fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        probabilities = self.models[model_name].predict_proba(X_scaled)
        class_labels = self.label_encoder.inverse_transform(self.models[model_name].classes_)
        
        return pd.DataFrame(probabilities, columns=class_labels, index=X.index)


class AnomalyDetector:
    """Detect anomalous vibration patterns."""
    
    def __init__(self, config: AlternativeFormulationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scaler = None
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit anomaly detection models."""
        self.logger.info("Training anomaly detection models")
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Define anomaly detectors
        detectors = {
            'isolation_forest': IsolationForest(
                contamination=self.config.isolation_forest_contamination,
                random_state=42,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=self.config.one_class_svm_nu,
                kernel='rbf',
                gamma='scale'
            ),
            'dbscan': DBSCAN(
                eps=self.config.dbscan_eps,
                min_samples=self.config.dbscan_min_samples
            )
        }
        
        results = {}
        
        for name, detector in detectors.items():
            self.logger.info(f"Training {name}...")
            
            try:
                if name == 'dbscan':
                    # DBSCAN doesn't have fit/predict_anomalies, use labels
                    labels = detector.fit_predict(X_scaled)
                    anomaly_labels = (labels == -1).astype(int)  # -1 indicates anomaly
                    self.models[name] = detector
                    
                    anomaly_rate = anomaly_labels.mean()
                    results[name] = {
                        'anomaly_rate': anomaly_rate,
                        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
                    }
                    
                else:
                    # Standard anomaly detectors
                    detector.fit(X_scaled)
                    anomaly_labels = detector.predict(X_scaled)
                    anomaly_labels = (anomaly_labels == -1).astype(int)  # Convert to 0/1
                    
                    self.models[name] = detector
                    
                    anomaly_rate = anomaly_labels.mean()
                    results[name] = {'anomaly_rate': anomaly_rate}
                
                self.logger.info(f"{name} - Anomaly rate: {anomaly_rate:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.fitted = True
        return results
    
    def predict_anomalies(self, X: pd.DataFrame, model_name: str = 'isolation_forest') -> pd.Series:
        """Predict anomalies in new data."""
        if not self.fitted:
            raise ValueError("Anomaly detector must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if model_name == 'dbscan':
            # DBSCAN requires re-fitting on new data
            labels = self.models[model_name].fit_predict(X_scaled)
            anomalies = (labels == -1).astype(int)
        else:
            predictions = self.models[model_name].predict(X_scaled)
            anomalies = (predictions == -1).astype(int)
        
        return pd.Series(anomalies, index=X.index)
    
    def get_anomaly_scores(self, X: pd.DataFrame, model_name: str = 'isolation_forest') -> pd.Series:
        """Get anomaly scores (if supported by the model)."""
        if not self.fitted:
            raise ValueError("Anomaly detector must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.models[model_name], 'score_samples'):
            scores = self.models[model_name].score_samples(X_scaled)
            # Convert to anomaly scores (higher = more anomalous)
            scores = -scores
        elif hasattr(self.models[model_name], 'decision_function'):
            scores = -self.models[model_name].decision_function(X_scaled)
        else:
            # Fallback: binary predictions
            scores = self.predict_anomalies(X, model_name).astype(float)
        
        return pd.Series(scores, index=X.index)


class ThresholdMonitor:
    """Monitor vibration thresholds and predict exceedance probability."""
    
    def __init__(self, config: AlternativeFormulationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.fitted = False
        
    def _create_threshold_targets(self, y: pd.Series) -> Dict[str, pd.Series]:
        """Create binary targets for threshold exceedance."""
        threshold_targets = {}
        
        for threshold_name, threshold_value in self.config.operational_thresholds.items():
            target = (y > threshold_value).astype(int)
            threshold_targets[threshold_name] = target
            
            exceedance_rate = target.mean()
            self.logger.info(f"Threshold {threshold_name} ({threshold_value}): {exceedance_rate:.4f} exceedance rate")
        
        return threshold_targets
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit threshold monitoring models."""
        self.logger.info("Training threshold monitoring models")
        
        # Create threshold targets
        threshold_targets = self._create_threshold_targets(y)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        self.scalers['main'] = scaler
        
        results = {}
        
        for threshold_name, target in threshold_targets.items():
            self.logger.info(f"Training model for {threshold_name} threshold...")
            
            # Skip if no positive examples
            if target.sum() == 0:
                self.logger.warning(f"No exceedances for {threshold_name} threshold. Skipping.")
                continue
            
            # Logistic regression for probability estimation
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            
            try:
                # Cross-validation
                cv = StratifiedKFold(n_splits=min(5, target.sum()), shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_scaled, target, cv=cv, scoring='roc_auc')
                
                # Fit full model
                model.fit(X_scaled, target)
                self.models[threshold_name] = model
                
                results[threshold_name] = {
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std(),
                    'exceedance_rate': target.mean()
                }
                
                self.logger.info(f"{threshold_name} - CV AUC: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {threshold_name} model: {e}")
                results[threshold_name] = {'error': str(e)}
        
        self.fitted = True
        return results
    
    def predict_exceedance_probability(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict probability of threshold exceedance."""
        if not self.fitted:
            raise ValueError("ThresholdMonitor must be fitted before prediction")
        
        X_scaled = pd.DataFrame(
            self.scalers['main'].transform(X),
            columns=X.columns,
            index=X.index
        )
        
        probabilities = {}
        
        for threshold_name, model in self.models.items():
            prob = model.predict_proba(X_scaled)[:, 1]  # Probability of exceedance (class 1)
            probabilities[f'prob_exceed_{threshold_name}'] = prob
        
        return pd.DataFrame(probabilities, index=X.index)
    
    def create_alert_system(self, exceedance_probs: pd.DataFrame) -> pd.DataFrame:
        """Create alert levels based on exceedance probabilities."""
        alerts = pd.DataFrame(index=exceedance_probs.index)
        
        # Define alert levels
        alert_thresholds = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        
        for threshold_name in self.config.operational_thresholds.keys():
            prob_col = f'prob_exceed_{threshold_name}'
            
            if prob_col in exceedance_probs.columns:
                prob_values = exceedance_probs[prob_col]
                
                alert_level = pd.Series('none', index=prob_values.index)
                alert_level[prob_values > alert_thresholds['low']] = 'low'
                alert_level[prob_values > alert_thresholds['medium']] = 'medium'
                alert_level[prob_values > alert_thresholds['high']] = 'high'
                
                alerts[f'alert_{threshold_name}'] = alert_level
        
        return alerts


class ChangePointDetector:
    """Detect significant changes in vibration patterns."""
    
    def __init__(self, config: AlternativeFormulationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.baseline_stats = {}
        self.fitted = False
        
    def fit(self, y: pd.Series) -> Dict[str, Any]:
        """Fit change detection model using baseline statistics."""
        self.logger.info("Fitting change point detection")
        
        # Calculate baseline statistics
        window = self.config.change_detection_window
        
        self.baseline_stats = {
            'mean': y.rolling(window=window*2, min_periods=window).mean().median(),
            'std': y.rolling(window=window*2, min_periods=window).std().median(),
            'median': y.rolling(window=window*2, min_periods=window).median().median()
        }
        
        self.logger.info(f"Baseline statistics: {self.baseline_stats}")
        self.fitted = True
        
        return {'baseline_stats': self.baseline_stats}
    
    def detect_changes(self, y: pd.Series) -> pd.DataFrame:
        """Detect change points in time series."""
        if not self.fitted:
            raise ValueError("ChangePointDetector must be fitted before detection")
        
        window = self.config.change_detection_window
        threshold_std = self.config.change_threshold_std
        
        results = pd.DataFrame(index=y.index)
        
        # Rolling statistics
        rolling_mean = y.rolling(window=window, min_periods=window//2).mean()
        rolling_std = y.rolling(window=window, min_periods=window//2).std()
        
        # Mean change detection
        mean_deviation = abs(rolling_mean - self.baseline_stats['mean'])
        mean_threshold = threshold_std * self.baseline_stats['std']
        results['mean_change'] = (mean_deviation > mean_threshold).astype(int)
        
        # Variance change detection
        std_ratio = rolling_std / self.baseline_stats['std']
        results['variance_change'] = ((std_ratio > 2.0) | (std_ratio < 0.5)).astype(int)
        
        # Combined change signal
        results['any_change'] = (results['mean_change'] | results['variance_change']).astype(int)
        
        # Change magnitude
        results['change_magnitude'] = mean_deviation / (self.baseline_stats['std'] + 1e-6)
        
        return results


class AlternativeFormulations:
    """Main class combining all alternative formulations."""
    
    def __init__(self, config: Optional[AlternativeFormulationConfig] = None):
        self.config = config or AlternativeFormulationConfig()
        self.logger = logging.getLogger(__name__)
        
        self.classifier = VibrationClassifier(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.threshold_monitor = ThresholdMonitor(self.config)
        self.change_detector = ChangePointDetector(self.config)
        
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Fit all alternative formulation models."""
        self.logger.info("Training all alternative formulation models")
        
        results = {}
        
        # Classification
        try:
            results['classification'] = self.classifier.fit(X, y)
            self.logger.info("Classification models trained successfully")
        except Exception as e:
            self.logger.error(f"Classification training failed: {e}")
            results['classification'] = {'error': str(e)}
        
        # Anomaly detection
        try:
            results['anomaly_detection'] = self.anomaly_detector.fit(X, y)
            self.logger.info("Anomaly detection models trained successfully")
        except Exception as e:
            self.logger.error(f"Anomaly detection training failed: {e}")
            results['anomaly_detection'] = {'error': str(e)}
        
        # Threshold monitoring
        try:
            results['threshold_monitoring'] = self.threshold_monitor.fit(X, y)
            self.logger.info("Threshold monitoring models trained successfully")
        except Exception as e:
            self.logger.error(f"Threshold monitoring training failed: {e}")
            results['threshold_monitoring'] = {'error': str(e)}
        
        # Change detection
        try:
            results['change_detection'] = self.change_detector.fit(y)
            self.logger.info("Change detection model trained successfully")
        except Exception as e:
            self.logger.error(f"Change detection training failed: {e}")
            results['change_detection'] = {'error': str(e)}
        
        self.fitted = True
        return results
    
    def predict_all(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Generate predictions from all formulations."""
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")
        
        results = {}
        
        # Classification predictions
        try:
            results['vibration_class'] = self.classifier.predict(X)
            results['class_probabilities'] = self.classifier.predict_proba(X)
        except Exception as e:
            self.logger.error(f"Classification prediction failed: {e}")
        
        # Anomaly detection
        try:
            results['anomaly_labels'] = self.anomaly_detector.predict_anomalies(X)
            results['anomaly_scores'] = self.anomaly_detector.get_anomaly_scores(X)
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
        
        # Threshold monitoring
        try:
            results['exceedance_probabilities'] = self.threshold_monitor.predict_exceedance_probability(X)
            results['alert_levels'] = self.threshold_monitor.create_alert_system(results['exceedance_probabilities'])
        except Exception as e:
            self.logger.error(f"Threshold monitoring failed: {e}")
        
        # Change detection (requires target values)
        if y is not None:
            try:
                results['change_points'] = self.change_detector.detect_changes(y)
            except Exception as e:
                self.logger.error(f"Change detection failed: {e}")
        
        return results


def create_alternative_config(
    vibration_thresholds: Optional[List[float]] = None,
    operational_thresholds: Optional[Dict[str, float]] = None
) -> AlternativeFormulationConfig:
    """Create configuration for alternative formulations."""
    if vibration_thresholds is None:
        vibration_thresholds = [4.5, 6.5, 8.5]  # Low, Normal, High, Critical boundaries
    
    if operational_thresholds is None:
        operational_thresholds = {
            'warning': 6.5,
            'alarm': 8.0,
            'emergency': 9.5
        }
    
    return AlternativeFormulationConfig(
        vibration_bins=vibration_thresholds,
        operational_thresholds=operational_thresholds,
        balance_classes=True,
        isolation_forest_contamination=0.1,
        change_detection_window=20
    )