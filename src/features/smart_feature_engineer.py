"""
Smart feature engineering pipeline with dynamic selection based on correlation analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings

from src.data.intelligent_analyzer import IntelligentDataAnalyzer
from config.settings import get_config


class SmartFeatureEngineer:
    """Smart feature engineer that dynamically creates and selects features."""
    
    def __init__(self):
        """Initialize the smart feature engineer."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.analyzer = IntelligentDataAnalyzer()
        
        self.selected_base_features = []
        self.engineered_features = []
        self.feature_importance_scores = {}
        self.is_fitted = False
        
        # Feature engineering parameters (will be optimized)
        self.optimal_lags = []
        self.optimal_windows = []
        self.temporal_patterns = {}
    
    def analyze_and_select_features(
        self,
        df: pd.DataFrame,
        target_column: str,
        correlation_threshold: float = 0.3,
        max_base_features: int = 15
    ) -> List[str]:
        """Analyze correlations and select the best base features.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            correlation_threshold: Minimum correlation threshold
            max_base_features: Maximum number of base features
            
        Returns:
            List of selected feature names
        """
        self.logger.info("Analyzing correlations for intelligent feature selection")
        
        # Use the intelligent analyzer
        selected_features = self.analyzer.select_features_intelligently(
            df, target_column, correlation_threshold, max_base_features
        )
        
        # Store results
        self.selected_base_features = selected_features
        self.correlation_results = self.analyzer.correlation_results
        
        # Analyze temporal patterns for feature engineering
        self.temporal_patterns = self.analyzer.analyze_time_series_patterns(
            df, target_column, selected_features[:5]  # Analyze top 5 features
        )
        
        self.logger.info(f"Selected {len(selected_features)} base features for engineering")
        return selected_features
    
    def determine_optimal_lags(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_column: str,
        max_lag: int = 20
    ) -> Dict[str, List[int]]:
        """Determine optimal lag values for each feature.
        
        Args:
            df: Input DataFrame
            features: List of feature names
            target_column: Target column name
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary mapping features to optimal lag values
        """
        self.logger.info("Determining optimal lag values for features")
        
        optimal_lags = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            feature_lags = []
            lag_scores = {}
            
            # Test different lag values
            for lag in range(1, min(max_lag + 1, len(df) // 10)):
                try:
                    # Create lagged feature
                    lagged_feature = df[feature].shift(lag)
                    
                    # Remove NaN values for correlation calculation
                    valid_data = pd.DataFrame({
                        'feature': lagged_feature,
                        'target': df[target_column]
                    }).dropna()
                    
                    if len(valid_data) < 100:  # Need sufficient data
                        continue
                    
                    # Calculate correlation
                    correlation = valid_data['feature'].corr(valid_data['target'])
                    
                    if not np.isnan(correlation) and abs(correlation) > 0.1:
                        lag_scores[lag] = abs(correlation)
                        
                except Exception as e:
                    self.logger.warning(f"Error testing lag {lag} for {feature}: {e}")
                    continue
            
            # Select top lags for this feature
            if lag_scores:
                # Sort by correlation and take top lags
                sorted_lags = sorted(lag_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Select lags with correlation > 0.15 or top 3
                good_lags = [lag for lag, score in sorted_lags if score > 0.15][:3]
                if not good_lags and sorted_lags:
                    good_lags = [sorted_lags[0][0]]  # At least take the best one
                
                optimal_lags[feature] = good_lags
                self.logger.debug(f"Optimal lags for {feature}: {good_lags}")
        
        self.optimal_lags = optimal_lags
        return optimal_lags
    
    def determine_optimal_windows(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_column: str
    ) -> Dict[str, List[int]]:
        """Determine optimal rolling window sizes for features.
        
        Args:
            df: Input DataFrame
            features: List of feature names
            target_column: Target column name
            
        Returns:
            Dictionary mapping features to optimal window sizes
        """
        self.logger.info("Determining optimal rolling window sizes")
        
        # Test different window sizes
        candidate_windows = [5, 10, 15, 30, 60, 120]  # Different time horizons
        optimal_windows = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            window_scores = {}
            
            for window in candidate_windows:
                if window >= len(df) // 5:  # Window too large
                    continue
                
                try:
                    # Test rolling mean
                    rolling_mean = df[feature].rolling(window=window, min_periods=1).mean()
                    
                    # Calculate correlation with target
                    valid_data = pd.DataFrame({
                        'feature': rolling_mean,
                        'target': df[target_column]
                    }).dropna()
                    
                    if len(valid_data) < 100:
                        continue
                    
                    correlation = valid_data['feature'].corr(valid_data['target'])
                    
                    if not np.isnan(correlation):
                        window_scores[window] = abs(correlation)
                        
                except Exception as e:
                    self.logger.warning(f"Error testing window {window} for {feature}: {e}")
                    continue
            
            # Select best windows
            if window_scores:
                sorted_windows = sorted(window_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Select windows that improve correlation or top 2
                base_corr = self.correlation_results['all_correlations'].loc[feature, 'abs_pearson'] \
                    if feature in self.correlation_results['all_correlations'].index else 0
                
                good_windows = [w for w, score in sorted_windows if score > base_corr * 1.05][:2]
                if not good_windows and sorted_windows:
                    good_windows = [sorted_windows[0][0]]
                
                optimal_windows[feature] = good_windows
                self.logger.debug(f"Optimal windows for {feature}: {good_windows}")
        
        self.optimal_windows = optimal_windows
        return optimal_windows
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        lag_dict: Optional[Dict[str, List[int]]] = None
    ) -> pd.DataFrame:
        """Create lag features for specified features.
        
        Args:
            df: Input DataFrame
            features: List of features to create lags for
            lag_dict: Dictionary of features to lag values (if None, use optimal_lags)
            
        Returns:
            DataFrame with lag features added
        """
        if lag_dict is None:
            lag_dict = self.optimal_lags
        
        df_with_lags = df.copy()
        created_features = []
        
        for feature in features:
            if feature not in df.columns or feature not in lag_dict:
                continue
            
            for lag in lag_dict[feature]:
                lag_feature_name = f"{feature}_lag_{lag}"
                df_with_lags[lag_feature_name] = df[feature].shift(lag)
                created_features.append(lag_feature_name)
        
        self.logger.info(f"Created {len(created_features)} lag features")
        return df_with_lags
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        window_dict: Optional[Dict[str, List[int]]] = None,
        statistics: List[str] = None
    ) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            df: Input DataFrame
            features: List of features to create rolling features for
            window_dict: Dictionary of features to window sizes
            statistics: List of statistics to compute
            
        Returns:
            DataFrame with rolling features added
        """
        if window_dict is None:
            window_dict = self.optimal_windows
        
        if statistics is None:
            statistics = ['mean', 'std']
        
        df_with_rolling = df.copy()
        created_features = []
        
        for feature in features:
            if feature not in df.columns or feature not in window_dict:
                continue
            
            for window in window_dict[feature]:
                for stat in statistics:
                    rolling_feature_name = f"{feature}_rolling_{stat}_{window}"
                    
                    if stat == 'mean':
                        df_with_rolling[rolling_feature_name] = df[feature].rolling(
                            window=window, min_periods=1
                        ).mean()
                    elif stat == 'std':
                        df_with_rolling[rolling_feature_name] = df[feature].rolling(
                            window=window, min_periods=1
                        ).std()
                    elif stat == 'min':
                        df_with_rolling[rolling_feature_name] = df[feature].rolling(
                            window=window, min_periods=1
                        ).min()
                    elif stat == 'max':
                        df_with_rolling[rolling_feature_name] = df[feature].rolling(
                            window=window, min_periods=1
                        ).max()
                    
                    created_features.append(rolling_feature_name)
        
        self.logger.info(f"Created {len(created_features)} rolling features")
        return df_with_rolling
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from the datetime index.
        
        Args:
            df: Input DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with temporal features added
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame does not have DatetimeIndex, skipping temporal features")
            return df
        
        df_with_temporal = df.copy()
        
        # Cyclical encoding for time features
        # Hour (0-23)
        df_with_temporal['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df_with_temporal['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # Day of week (0-6)
        df_with_temporal['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df_with_temporal['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Day of month (1-31)
        df_with_temporal['day_of_month_sin'] = np.sin(2 * np.pi * (df.index.day - 1) / 31)
        df_with_temporal['day_of_month_cos'] = np.cos(2 * np.pi * (df.index.day - 1) / 31)
        
        # Additional features based on temporal patterns
        if self.temporal_patterns:
            # Weekend indicator
            df_with_temporal['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Business hours indicator (assuming 8-17)
            df_with_temporal['is_business_hours'] = (
                (df.index.hour >= 8) & (df.index.hour <= 17)
            ).astype(int)
        
        self.logger.info("Created temporal features")
        return df_with_temporal
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        max_interactions: int = 5
    ) -> pd.DataFrame:
        """Create interaction features between top correlated features.
        
        Args:
            df: Input DataFrame
            features: List of features to create interactions for
            max_interactions: Maximum number of interaction features
            
        Returns:
            DataFrame with interaction features added
        """
        df_with_interactions = df.copy()
        
        # Select top features for interactions
        if len(features) < 2:
            return df_with_interactions
        
        top_features = features[:min(4, len(features))]  # Use top 4 for interactions
        interactions_created = 0
        
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features[i+1:], i+1):
                if interactions_created >= max_interactions:
                    break
                
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplication interaction
                    interaction_name = f"{feat1}_x_{feat2}"
                    df_with_interactions[interaction_name] = df[feat1] * df[feat2]
                    interactions_created += 1
        
        if interactions_created > 0:
            self.logger.info(f"Created {interactions_created} interaction features")
        
        return df_with_interactions
    
    def fit(self, df: pd.DataFrame, target_column: str) -> 'SmartFeatureEngineer':
        """Fit the smart feature engineer.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Fitting smart feature engineer")
        
        # Step 1: Analyze and select base features
        self.selected_base_features = self.analyze_and_select_features(
            df, target_column, 
            correlation_threshold=0.25,  # Slightly lower threshold for more features
            max_base_features=15
        )
        
        if not self.selected_base_features:
            self.logger.error("No features selected. Check data quality.")
            return self
        
        # Step 2: Determine optimal parameters for feature engineering
        self.determine_optimal_lags(df, self.selected_base_features[:8], target_column, max_lag=10)
        self.determine_optimal_windows(df, self.selected_base_features[:8], target_column)
        
        self.is_fitted = True
        self.logger.info("Smart feature engineer fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame using fitted parameters.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame with engineered features
        """
        if not self.is_fitted:
            raise ValueError("SmartFeatureEngineer must be fitted before transform")
        
        self.logger.info("Transforming data with smart feature engineering")
        
        # Start with selected base features
        available_features = [f for f in self.selected_base_features if f in df.columns]
        
        if not available_features:
            self.logger.error("No selected features available in DataFrame")
            return df
        
        # Create base feature set
        df_engineered = df[available_features].copy()
        
        # Add temporal features
        if isinstance(df.index, pd.DatetimeIndex):
            df_temp = df.copy()
            df_temp = self.create_temporal_features(df_temp)
            
            # Add temporal features to engineered dataframe
            temporal_cols = [col for col in df_temp.columns if col not in df.columns]
            for col in temporal_cols:
                df_engineered[col] = df_temp[col]
        
        # Create lag features
        if self.optimal_lags:
            df_engineered = self.create_lag_features(
                df_engineered, available_features, self.optimal_lags
            )
        
        # Create rolling features  
        if self.optimal_windows:
            df_engineered = self.create_rolling_features(
                df_engineered, available_features[:6], self.optimal_windows, ['mean', 'std']
            )
        
        # Create interaction features
        df_engineered = self.create_interaction_features(
            df_engineered, available_features, max_interactions=3
        )
        
        self.logger.info(f"Feature engineering completed: {df.shape[1]} -> {df_engineered.shape[1]} features")
        
        return df_engineered
    
    def fit_transform(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Fit and transform the DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            
        Returns:
            Transformed DataFrame with engineered features
        """
        return self.fit(df, target_column).transform(df)
    
    def get_feature_importance_report(self) -> Dict[str, Any]:
        """Get a report of feature importance and engineering decisions.
        
        Returns:
            Dictionary with feature engineering report
        """
        if not self.is_fitted:
            return {}
        
        report = {
            'selected_base_features': self.selected_base_features,
            'correlation_summary': {
                'total_features_analyzed': len(self.correlation_results.get('all_correlations', [])),
                'features_above_threshold': len(self.correlation_results.get('significant_features', [])),
                'selected_features': len(self.selected_base_features)
            },
            'lag_features': {
                'features_with_lags': list(self.optimal_lags.keys()),
                'total_lag_features': sum(len(lags) for lags in self.optimal_lags.values())
            },
            'rolling_features': {
                'features_with_rolling': list(self.optimal_windows.keys()),
                'total_rolling_features': sum(len(windows) * 2 for windows in self.optimal_windows.values())  # 2 stats per window
            },
            'optimal_parameters': {
                'optimal_lags': self.optimal_lags,
                'optimal_windows': self.optimal_windows
            }
        }
        
        return report


def create_smart_feature_engineer() -> SmartFeatureEngineer:
    """Create a smart feature engineer instance."""
    return SmartFeatureEngineer()