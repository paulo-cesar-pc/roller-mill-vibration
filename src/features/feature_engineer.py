"""
Advanced feature engineering for the roller mill vibration prediction project.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings

from config.settings import get_config


class BaseFeatureEngineer(ABC):
    """Base class for feature engineering components."""
    
    def __init__(self, name: str):
        """Initialize the feature engineer.
        
        Args:
            name: Name of the feature engineer.
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'BaseFeatureEngineer':
        """Fit the feature engineer to the data.
        
        Args:
            df: Input DataFrame.
            target: Target series (if needed).
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Transformed DataFrame.
        """
        pass
    
    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the DataFrame.
        
        Args:
            df: Input DataFrame.
            target: Target series (if needed).
            
        Returns:
            Transformed DataFrame.
        """
        return self.fit(df, target).transform(df)


class LagFeatureEngineer(BaseFeatureEngineer):
    """Creates lag features for time series data."""
    
    def __init__(self, columns: List[str], lags: List[int]):
        """Initialize the lag feature engineer.
        
        Args:
            columns: Columns to create lag features for.
            lags: List of lag values to create.
        """
        super().__init__("LagFeatureEngineer")
        self.columns = columns
        self.lags = lags
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'LagFeatureEngineer':
        """Fit the lag feature engineer."""
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features."""
        if not self.is_fitted:
            raise ValueError("LagFeatureEngineer must be fitted before transform")
        
        df_result = df.copy()
        
        for column in self.columns:
            if column not in df.columns:
                self.logger.warning(f"Column '{column}' not found, skipping lag features")
                continue
            
            for lag in self.lags:
                lag_column_name = f"{column}_lag_{lag}"
                df_result[lag_column_name] = df[column].shift(lag)
                
        self.logger.info(f"Created {len(self.columns) * len(self.lags)} lag features")
        return df_result


class RollingFeatureEngineer(BaseFeatureEngineer):
    """Creates rolling window features for time series data."""
    
    def __init__(self, columns: List[str], windows: List[int], statistics: List[str]):
        """Initialize the rolling feature engineer.
        
        Args:
            columns: Columns to create rolling features for.
            windows: List of window sizes.
            statistics: List of statistics to compute ('mean', 'std', 'min', 'max', 'median').
        """
        super().__init__("RollingFeatureEngineer")
        self.columns = columns
        self.windows = windows
        self.statistics = statistics
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'RollingFeatureEngineer':
        """Fit the rolling feature engineer."""
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features."""
        if not self.is_fitted:
            raise ValueError("RollingFeatureEngineer must be fitted before transform")
        
        df_result = df.copy()
        
        for column in self.columns:
            if column not in df.columns:
                self.logger.warning(f"Column '{column}' not found, skipping rolling features")
                continue
            
            for window in self.windows:
                rolling = df[column].rolling(window=window, min_periods=1)
                
                for stat in self.statistics:
                    stat_column_name = f"{column}_rolling_{stat}_{window}"
                    
                    if stat == 'mean':
                        df_result[stat_column_name] = rolling.mean()
                    elif stat == 'std':
                        df_result[stat_column_name] = rolling.std()
                    elif stat == 'min':
                        df_result[stat_column_name] = rolling.min()
                    elif stat == 'max':
                        df_result[stat_column_name] = rolling.max()
                    elif stat == 'median':
                        df_result[stat_column_name] = rolling.median()
                    elif stat == 'skew':
                        df_result[stat_column_name] = rolling.skew()
                    elif stat == 'kurt':
                        df_result[stat_column_name] = rolling.kurt()
        
        feature_count = len(self.columns) * len(self.windows) * len(self.statistics)
        self.logger.info(f"Created {feature_count} rolling window features")
        return df_result


class TemporalFeatureEngineer(BaseFeatureEngineer):
    """Creates temporal features from datetime index."""
    
    def __init__(self, cyclical: bool = True, include_features: List[str] = None):
        """Initialize the temporal feature engineer.
        
        Args:
            cyclical: Whether to create cyclical encodings for temporal features.
            include_features: List of temporal features to include.
        """
        super().__init__("TemporalFeatureEngineer")
        self.cyclical = cyclical
        self.include_features = include_features or ['hour', 'day_of_week', 'day_of_month', 'month']
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'TemporalFeatureEngineer':
        """Fit the temporal feature engineer."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for temporal features")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features."""
        if not self.is_fitted:
            raise ValueError("TemporalFeatureEngineer must be fitted before transform")
        
        df_result = df.copy()
        
        # Extract basic temporal features
        if 'hour' in self.include_features:
            df_result['hour'] = df.index.hour
            if self.cyclical:
                df_result['hour_sin'] = np.sin(2 * np.pi * df_result['hour'] / 24)
                df_result['hour_cos'] = np.cos(2 * np.pi * df_result['hour'] / 24)
                df_result = df_result.drop(columns=['hour'])
        
        if 'day_of_week' in self.include_features:
            df_result['day_of_week'] = df.index.dayofweek
            if self.cyclical:
                df_result['day_of_week_sin'] = np.sin(2 * np.pi * df_result['day_of_week'] / 7)
                df_result['day_of_week_cos'] = np.cos(2 * np.pi * df_result['day_of_week'] / 7)
                df_result = df_result.drop(columns=['day_of_week'])
        
        if 'day_of_month' in self.include_features:
            df_result['day_of_month'] = df.index.day
            if self.cyclical:
                df_result['day_of_month_sin'] = np.sin(2 * np.pi * df_result['day_of_month'] / 31)
                df_result['day_of_month_cos'] = np.cos(2 * np.pi * df_result['day_of_month'] / 31)
                df_result = df_result.drop(columns=['day_of_month'])
        
        if 'month' in self.include_features:
            df_result['month'] = df.index.month
            if self.cyclical:
                df_result['month_sin'] = np.sin(2 * np.pi * df_result['month'] / 12)
                df_result['month_cos'] = np.cos(2 * np.pi * df_result['month'] / 12)
                df_result = df_result.drop(columns=['month'])
        
        # Additional temporal features
        if 'is_weekend' in self.include_features:
            df_result['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        if 'quarter' in self.include_features:
            df_result['quarter'] = df.index.quarter
        
        self.logger.info(f"Created temporal features for {len(self.include_features)} time components")
        return df_result


class SpectralFeatureEngineer(BaseFeatureEngineer):
    """Creates spectral features using FFT analysis."""
    
    def __init__(self, columns: List[str], n_components: int = 10, window_size: int = 100):
        """Initialize the spectral feature engineer.
        
        Args:
            columns: Columns to create spectral features for.
            n_components: Number of FFT components to keep.
            window_size: Window size for rolling FFT analysis.
        """
        super().__init__("SpectralFeatureEngineer")
        self.columns = columns
        self.n_components = n_components
        self.window_size = window_size
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'SpectralFeatureEngineer':
        """Fit the spectral feature engineer."""
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spectral features."""
        if not self.is_fitted:
            raise ValueError("SpectralFeatureEngineer must be fitted before transform")
        
        df_result = df.copy()
        
        for column in self.columns:
            if column not in df.columns:
                self.logger.warning(f"Column '{column}' not found, skipping spectral features")
                continue
            
            series = df[column].fillna(method='ffill').fillna(method='bfill')
            
            # Rolling FFT analysis
            for i in range(self.n_components):
                fft_feature = f"{column}_fft_{i}"
                df_result[fft_feature] = self._rolling_fft_component(series, i)
            
            # Spectral statistics
            df_result[f"{column}_spectral_energy"] = self._rolling_spectral_energy(series)
            df_result[f"{column}_spectral_centroid"] = self._rolling_spectral_centroid(series)
        
        feature_count = len(self.columns) * (self.n_components + 2)
        self.logger.info(f"Created {feature_count} spectral features")
        return df_result
    
    def _rolling_fft_component(self, series: pd.Series, component: int) -> pd.Series:
        """Compute rolling FFT component."""
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            start_idx = max(0, i - self.window_size + 1)
            window_data = series.iloc[start_idx:i+1]
            
            if len(window_data) > 10:  # Minimum window size
                fft_values = np.abs(fft(window_data.values))
                if component < len(fft_values):
                    result.iloc[i] = fft_values[component]
                else:
                    result.iloc[i] = 0
            else:
                result.iloc[i] = 0
        
        return result
    
    def _rolling_spectral_energy(self, series: pd.Series) -> pd.Series:
        """Compute rolling spectral energy."""
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            start_idx = max(0, i - self.window_size + 1)
            window_data = series.iloc[start_idx:i+1]
            
            if len(window_data) > 10:
                fft_values = np.abs(fft(window_data.values))
                result.iloc[i] = np.sum(fft_values**2)
            else:
                result.iloc[i] = 0
        
        return result
    
    def _rolling_spectral_centroid(self, series: pd.Series) -> pd.Series:
        """Compute rolling spectral centroid."""
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(len(series)):
            start_idx = max(0, i - self.window_size + 1)
            window_data = series.iloc[start_idx:i+1]
            
            if len(window_data) > 10:
                fft_values = np.abs(fft(window_data.values))
                freqs = fftfreq(len(fft_values))
                
                # Avoid division by zero
                if np.sum(fft_values) > 0:
                    result.iloc[i] = np.sum(freqs * fft_values) / np.sum(fft_values)
                else:
                    result.iloc[i] = 0
            else:
                result.iloc[i] = 0
        
        return result


class SeasonalityFeatureEngineer(BaseFeatureEngineer):
    """Creates seasonality features using time series decomposition."""
    
    def __init__(self, columns: List[str], periods: List[int]):
        """Initialize the seasonality feature engineer.
        
        Args:
            columns: Columns to decompose.
            periods: List of seasonal periods to consider.
        """
        super().__init__("SeasonalityFeatureEngineer")
        self.columns = columns
        self.periods = periods
        self.seasonal_components = {}
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'SeasonalityFeatureEngineer':
        """Fit the seasonality feature engineer."""
        for column in self.columns:
            if column not in df.columns:
                continue
            
            self.seasonal_components[column] = {}
            series = df[column].fillna(method='ffill').fillna(method='bfill')
            
            for period in self.periods:
                if len(series) >= 2 * period:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            decomposition = seasonal_decompose(
                                series, 
                                model='additive', 
                                period=period,
                                extrapolate_trend='freq'
                            )
                            self.seasonal_components[column][period] = {
                                'trend': decomposition.trend,
                                'seasonal': decomposition.seasonal,
                                'residual': decomposition.resid
                            }
                    except Exception as e:
                        self.logger.warning(f"Could not decompose {column} with period {period}: {e}")
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonality features."""
        if not self.is_fitted:
            raise ValueError("SeasonalityFeatureEngineer must be fitted before transform")
        
        df_result = df.copy()
        
        for column, period_dict in self.seasonal_components.items():
            for period, components in period_dict.items():
                # Add trend component
                trend_name = f"{column}_trend_{period}"
                df_result[trend_name] = components['trend'].reindex(df.index, method='nearest')
                
                # Add seasonal component
                seasonal_name = f"{column}_seasonal_{period}"
                df_result[seasonal_name] = components['seasonal'].reindex(df.index, method='nearest')
                
                # Add residual component (noise)
                residual_name = f"{column}_residual_{period}"
                df_result[residual_name] = components['residual'].reindex(df.index, method='nearest')
        
        feature_count = len(self.seasonal_components) * len(self.periods) * 3
        self.logger.info(f"Created {feature_count} seasonality features")
        return df_result


class InteractionFeatureEngineer(BaseFeatureEngineer):
    """Creates interaction features between columns."""
    
    def __init__(self, interactions: List[Tuple[str, str]], operations: List[str] = None):
        """Initialize the interaction feature engineer.
        
        Args:
            interactions: List of column pairs to create interactions for.
            operations: List of operations ('multiply', 'divide', 'add', 'subtract').
        """
        super().__init__("InteractionFeatureEngineer")
        self.interactions = interactions
        self.operations = operations or ['multiply', 'divide']
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'InteractionFeatureEngineer':
        """Fit the interaction feature engineer."""
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        if not self.is_fitted:
            raise ValueError("InteractionFeatureEngineer must be fitted before transform")
        
        df_result = df.copy()
        
        for col1, col2 in self.interactions:
            if col1 not in df.columns or col2 not in df.columns:
                self.logger.warning(f"Columns '{col1}' or '{col2}' not found, skipping interaction")
                continue
            
            for operation in self.operations:
                if operation == 'multiply':
                    feature_name = f"{col1}_x_{col2}"
                    df_result[feature_name] = df[col1] * df[col2]
                elif operation == 'divide':
                    feature_name = f"{col1}_div_{col2}"
                    # Avoid division by zero
                    df_result[feature_name] = df[col1] / (df[col2] + 1e-8)
                elif operation == 'add':
                    feature_name = f"{col1}_plus_{col2}"
                    df_result[feature_name] = df[col1] + df[col2]
                elif operation == 'subtract':
                    feature_name = f"{col1}_minus_{col2}"
                    df_result[feature_name] = df[col1] - df[col2]
        
        feature_count = len(self.interactions) * len(self.operations)
        self.logger.info(f"Created {feature_count} interaction features")
        return df_result


class CategoricalFeatureEngineer(BaseFeatureEngineer):
    """Handles categorical feature encoding."""
    
    def __init__(self, columns: List[str], method: str = 'onehot'):
        """Initialize the categorical feature engineer.
        
        Args:
            columns: Categorical columns to encode.
            method: Encoding method ('onehot', 'label', 'target').
        """
        super().__init__("CategoricalFeatureEngineer")
        self.columns = columns
        self.method = method
        self.encoders = {}
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'CategoricalFeatureEngineer':
        """Fit the categorical encoders."""
        for column in self.columns:
            if column not in df.columns:
                continue
            
            if self.method == 'label':
                encoder = LabelEncoder()
                encoder.fit(df[column].fillna('missing'))
                self.encoders[column] = encoder
            elif self.method == 'target' and target is not None:
                # Target encoding (mean encoding)
                target_means = df.groupby(column)[target.name].mean()
                global_mean = target.mean()
                self.encoders[column] = {'means': target_means, 'global_mean': global_mean}
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        if not self.is_fitted:
            raise ValueError("CategoricalFeatureEngineer must be fitted before transform")
        
        df_result = df.copy()
        
        for column in self.columns:
            if column not in df.columns:
                continue
            
            if self.method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
                df_result = pd.concat([df_result, dummies], axis=1)
                df_result = df_result.drop(columns=[column])
                
            elif self.method == 'label' and column in self.encoders:
                # Label encoding
                encoder = self.encoders[column]
                df_result[column] = encoder.transform(df[column].fillna('missing'))
                
            elif self.method == 'target' and column in self.encoders:
                # Target encoding
                means_dict = self.encoders[column]['means']
                global_mean = self.encoders[column]['global_mean']
                df_result[column] = df[column].map(means_dict).fillna(global_mean)
        
        self.logger.info(f"Encoded {len(self.columns)} categorical features using {self.method} method")
        return df_result


class FeatureSelector(BaseFeatureEngineer):
    """Selects the best features using various methods."""
    
    def __init__(self, method: str = 'mutual_info', k: int = 50):
        """Initialize the feature selector.
        
        Args:
            method: Selection method ('mutual_info', 'f_regression', 'correlation').
            k: Number of features to select.
        """
        super().__init__("FeatureSelector")
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
    
    def fit(self, df: pd.DataFrame, target: pd.Series) -> 'FeatureSelector':
        """Fit the feature selector."""
        if target is None:
            raise ValueError("Target is required for feature selection")
        
        # Remove non-numeric columns for feature selection
        numeric_df = df.select_dtypes(include=[np.number])
        
        if self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_regression, k=min(self.k, numeric_df.shape[1]))
        elif self.method == 'f_regression':
            self.selector = SelectKBest(score_func=f_regression, k=min(self.k, numeric_df.shape[1]))
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
        
        # Align target with DataFrame
        aligned_target = target.reindex(numeric_df.index)
        
        # Remove rows with NaN in target
        valid_mask = ~aligned_target.isna()
        numeric_df_clean = numeric_df[valid_mask]
        aligned_target_clean = aligned_target[valid_mask]
        
        # Fit selector
        self.selector.fit(numeric_df_clean.fillna(0), aligned_target_clean)
        
        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = numeric_df.columns[selected_indices].tolist()
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the best features."""
        if not self.is_fitted:
            raise ValueError("FeatureSelector must be fitted before transform")
        
        # Keep only selected numeric features and all non-numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        non_numeric_df = df.select_dtypes(exclude=[np.number])
        
        selected_numeric = numeric_df[self.selected_features]
        
        df_result = pd.concat([selected_numeric, non_numeric_df], axis=1)
        
        self.logger.info(f"Selected {len(self.selected_features)} features out of {len(numeric_df.columns)} numeric features")
        return df_result


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the feature engineering pipeline."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.engineers = []
        self.is_fitted = False
    
    def add_engineer(self, engineer: BaseFeatureEngineer) -> 'FeatureEngineeringPipeline':
        """Add a feature engineer to the pipeline.
        
        Args:
            engineer: Feature engineer to add.
            
        Returns:
            Self for method chaining.
        """
        self.engineers.append(engineer)
        return self
    
    def fit(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> 'FeatureEngineeringPipeline':
        """Fit all feature engineers in the pipeline.
        
        Args:
            df: Input DataFrame.
            target: Target series.
            
        Returns:
            Self for method chaining.
        """
        current_df = df.copy()
        
        for engineer in self.engineers:
            self.logger.info(f"Fitting {engineer.name}")
            engineer.fit(current_df, target)
            current_df = engineer.transform(current_df)
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the DataFrame using all engineers.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Transformed DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        current_df = df.copy()
        
        for engineer in self.engineers:
            self.logger.info(f"Applying {engineer.name}")
            current_df = engineer.transform(current_df)
        
        return current_df
    
    def fit_transform(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform the DataFrame.
        
        Args:
            df: Input DataFrame.
            target: Target series.
            
        Returns:
            Transformed DataFrame.
        """
        return self.fit(df, target).transform(df)


def create_default_pipeline(config) -> FeatureEngineeringPipeline:
    """Create a default feature engineering pipeline based on configuration.
    
    Args:
        config: Project configuration.
        
    Returns:
        Configured feature engineering pipeline.
    """
    pipeline = FeatureEngineeringPipeline()
    
    # Categorical encoding
    if config.features.categorical_columns:
        categorical_engineer = CategoricalFeatureEngineer(
            columns=config.features.categorical_columns,
            method='onehot'
        )
        pipeline.add_engineer(categorical_engineer)
    
    # Lag features
    if config.features.lag_features['columns']:
        lag_engineer = LagFeatureEngineer(
            columns=config.features.lag_features['columns'],
            lags=config.features.lag_features['lags']
        )
        pipeline.add_engineer(lag_engineer)
    
    # Rolling features
    if config.features.rolling_features['columns']:
        rolling_engineer = RollingFeatureEngineer(
            columns=config.features.rolling_features['columns'],
            windows=config.features.rolling_features['windows'],
            statistics=config.features.rolling_features['statistics']
        )
        pipeline.add_engineer(rolling_engineer)
    
    # Temporal features
    if config.features.temporal_features.get('enabled', True):
        include_features = []
        if config.features.temporal_features.get('hour', True):
            include_features.append('hour')
        if config.features.temporal_features.get('day_of_week', True):
            include_features.append('day_of_week')
        if config.features.temporal_features.get('day_of_month', False):
            include_features.append('day_of_month')
        if config.features.temporal_features.get('month', False):
            include_features.append('month')
        
        temporal_engineer = TemporalFeatureEngineer(
            cyclical=config.features.temporal_features.get('cyclical', True),
            include_features=include_features
        )
        pipeline.add_engineer(temporal_engineer)
    
    # Spectral features
    if config.features.spectral_features.get('enabled', False):
        spectral_engineer = SpectralFeatureEngineer(
            columns=config.features.high_corr_features[:3],  # Apply to top 3 features
            n_components=config.features.spectral_features.get('fft_components', 10)
        )
        pipeline.add_engineer(spectral_engineer)
    
    # Seasonality features
    if config.features.seasonality_features.get('enabled', False):
        seasonality_engineer = SeasonalityFeatureEngineer(
            columns=config.features.high_corr_features[:3],  # Apply to top 3 features
            periods=config.features.seasonality_features.get('periods', [24, 168])
        )
        pipeline.add_engineer(seasonality_engineer)
    
    # Interaction features
    high_corr_cols = config.features.high_corr_features
    if len(high_corr_cols) >= 2:
        interactions = [(high_corr_cols[0], high_corr_cols[1])]
        if len(high_corr_cols) >= 3:
            interactions.append((high_corr_cols[1], high_corr_cols[2]))
        
        interaction_engineer = InteractionFeatureEngineer(
            interactions=interactions,
            operations=['multiply', 'divide']
        )
        pipeline.add_engineer(interaction_engineer)
    
    return pipeline