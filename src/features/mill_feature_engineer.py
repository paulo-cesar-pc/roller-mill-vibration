"""
Process-aware feature engineering specifically designed for roller mill operations.
Creates domain-specific features that capture mill physics and operational patterns.
"""

import pandas as pd
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MillFeatureConfig:
    """Configuration for mill-specific feature engineering."""
    # Multi-scale windows (in number of aggregated samples)
    short_windows: List[int] = None
    medium_windows: List[int] = None
    long_windows: List[int] = None
    
    # Process-specific thresholds
    power_stability_threshold: float = 0.05
    speed_stability_threshold: float = 2.0
    pressure_change_threshold: float = 0.1
    
    # Feature selection
    create_efficiency_features: bool = True
    create_stability_features: bool = True
    create_interaction_features: bool = True
    create_change_features: bool = True
    create_ratio_features: bool = True
    
    def __post_init__(self):
        if self.short_windows is None:
            self.short_windows = [3, 6, 12]  # 15min, 30min, 1hr for 5min aggregation
        if self.medium_windows is None:
            self.medium_windows = [24, 48, 96]  # 2hr, 4hr, 8hr
        if self.long_windows is None:
            self.long_windows = [144, 288, 576]  # 12hr, 24hr, 48hr


class MillFeatureEngineer:
    """
    Specialized feature engineering for roller mill vibration prediction.
    Creates physics-based and operational features specific to mill operations.
    """
    
    def __init__(self, config: Optional[MillFeatureConfig] = None):
        """Initialize with mill-specific configuration."""
        self.config = config or MillFeatureConfig()
        self.logger = logging.getLogger(__name__)
        self.fitted = False
        self.feature_importance_map = {}
        
        # Define mill-specific column mappings
        self.mill_columns = {
            'power': 'CM2_PV_VRM01_POWER',
            'vibration': 'CM2_PV_VRM01_VIBRATION', 
            'speed': 'CM2_PV_CLA01_SPEED',
            'feed_rate': 'CM2_PV_RB01_TOTAL_FEED',
            'inlet_pressure': 'CM2_PV_VRM01_IN_PRESS',
            'outlet_pressure': 'CM2_PV_VRM01_OUT_ PRESS',
            'diff_pressure': 'CM2_PV_VRM01_DIFF_PRESSURE',
            'inlet_temp': 'CM2_PV_VRM01_INLET_TEMPERATURE',
            'outlet_temp': 'CM2_PV_VRM01_OUTLET_TEMPERATURE',
            'water_injection': 'CM2_PV_WI01_WATER_INJECTION',
            'product_type': 'CM2_PV_PRODUCT'
        }
        
    def create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create mill efficiency and performance indicators."""
        df_eff = df.copy()
        
        power_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_POWER', ['mean'])[0]
        feed_col = self._get_aggregated_cols(df, 'CM2_PV_RB01_TOTAL_FEED', ['mean'])[0] 
        speed_col = self._get_aggregated_cols(df, 'CM2_PV_CLA01_SPEED', ['mean'])[0]
        
        if all(col in df.columns for col in [power_col, feed_col, speed_col]):
            # Specific energy (power per unit of feed)
            df_eff['specific_energy'] = df[power_col] / (df[feed_col] + 1e-6)
            
            # Mill loading index (feed rate vs speed relationship) 
            df_eff['loading_index'] = df[feed_col] / (df[speed_col] + 1e-6)
            
            # Power efficiency (actual vs expected power)
            expected_power = 0.8 * df[feed_col] + 0.2 * df[speed_col]  # Simplified model
            df_eff['power_efficiency'] = df[power_col] / (expected_power + 1e-6)
            
            # Mill productivity index
            df_eff['productivity_index'] = df[feed_col] * df[speed_col] / (df[power_col] + 1e-6)
            
        self.logger.debug("Created efficiency features")
        return df_eff
    
    def create_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create operational stability indicators."""
        df_stab = df.copy()
        
        # Find standard deviation columns for key parameters
        power_std_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_POWER', ['std'])[0]
        speed_std_col = self._get_aggregated_cols(df, 'CM2_PV_CLA01_SPEED', ['std'])[0]
        feed_std_col = self._get_aggregated_cols(df, 'CM2_PV_RB01_TOTAL_FEED', ['std'])[0]
        
        power_mean_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_POWER', ['mean'])[0]
        speed_mean_col = self._get_aggregated_cols(df, 'CM2_PV_CLA01_SPEED', ['mean'])[0]
        feed_mean_col = self._get_aggregated_cols(df, 'CM2_PV_RB01_TOTAL_FEED', ['mean'])[0]
        
        if all(col in df.columns for col in [power_std_col, power_mean_col]):
            # Coefficient of variation for stability
            df_stab['power_stability'] = df[power_std_col] / (df[power_mean_col] + 1e-6)
            df_stab['power_stable_flag'] = (df_stab['power_stability'] < self.config.power_stability_threshold).astype(int)
            
        if all(col in df.columns for col in [speed_std_col, speed_mean_col]):
            df_stab['speed_stability'] = df[speed_std_col] / (df[speed_mean_col] + 1e-6)
            df_stab['speed_stable_flag'] = (df[speed_std_col] < self.config.speed_stability_threshold).astype(int)
            
        if all(col in df.columns for col in [feed_std_col, feed_mean_col]):
            df_stab['feed_stability'] = df[feed_std_col] / (df[feed_mean_col] + 1e-6)
            
        # Multi-parameter stability index
        stability_cols = [col for col in df_stab.columns if col.endswith('_stability')]
        if stability_cols:
            df_stab['overall_stability'] = df_stab[stability_cols].mean(axis=1)
            df_stab['stable_operation'] = (df_stab['overall_stability'] < 0.1).astype(int)
            
        self.logger.debug("Created stability features")
        return df_stab
    
    def create_process_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create process-specific ratio features."""
        df_ratio = df.copy()
        
        # Pressure ratios
        inlet_p_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_IN_PRESS', ['mean'])[0]
        outlet_p_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_OUT_ PRESS', ['mean'])[0]
        diff_p_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_DIFF_PRESSURE', ['mean'])[0]
        
        if all(col in df.columns for col in [inlet_p_col, outlet_p_col]):
            df_ratio['pressure_ratio'] = df[inlet_p_col] / (df[outlet_p_col] + 1e-6)
            df_ratio['pressure_drop_ratio'] = df[diff_p_col] / (df[inlet_p_col] + 1e-6)
            
        # Temperature ratios  
        inlet_t_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_INLET_TEMPERATURE', ['mean'])[0]
        outlet_t_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_OUTLET_TEMPERATURE', ['mean'])[0]
        
        if all(col in df.columns for col in [inlet_t_col, outlet_t_col]):
            df_ratio['temperature_rise'] = df[outlet_t_col] - df[inlet_t_col]
            df_ratio['temperature_ratio'] = df[outlet_t_col] / (df[inlet_t_col] + 273.15)  # Absolute temperature ratio
            
        # Water injection efficiency
        water_col = self._get_aggregated_cols(df, 'CM2_PV_WI01_WATER_INJECTION', ['mean'])[0]
        power_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_POWER', ['mean'])[0]
        
        if all(col in df.columns for col in [water_col, power_col]):
            df_ratio['water_to_power_ratio'] = df[water_col] / (df[power_col] + 1e-6)
            
        self.logger.debug("Created process ratio features")
        return df_ratio
    
    def create_change_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that detect operational changes."""
        df_change = df.copy()
        
        # Key parameters for change detection
        key_params = [
            'CM2_PV_VRM01_POWER_mean',
            'CM2_PV_CLA01_SPEED_mean', 
            'CM2_PV_RB01_TOTAL_FEED_mean',
            'CM2_PV_VRM01_DIFF_PRESSURE_mean'
        ]
        
        for param in key_params:
            if param in df.columns:
                base_name = param.replace('_mean', '')
                
                # Rate of change (derivative)
                df_change[f'{base_name}_rate_of_change'] = df[param].diff()
                
                # Acceleration (second derivative)
                df_change[f'{base_name}_acceleration'] = df[param].diff().diff()
                
                # Rolling change detection
                for window in self.config.short_windows:
                    df_change[f'{base_name}_change_{window}'] = df[param] - df[param].shift(window)
                    df_change[f'{base_name}_pct_change_{window}'] = df[param].pct_change(periods=window)
                    
                # Change magnitude
                df_change[f'{base_name}_abs_change_3'] = abs(df[param].diff(3))
                
                # Change direction persistence
                df_change[f'{base_name}_trend_3'] = np.sign(df[param].diff()).rolling(3).sum()
                
        self.logger.debug("Created change detection features")
        return df_change
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key mill parameters."""
        df_interact = df.copy()
        
        # Key parameter pairs for interactions
        interactions = [
            ('CM2_PV_VRM01_POWER_mean', 'CM2_PV_CLA01_SPEED_mean'),
            ('CM2_PV_RB01_TOTAL_FEED_mean', 'CM2_PV_VRM01_POWER_mean'),
            ('CM2_PV_VRM01_DIFF_PRESSURE_mean', 'CM2_PV_RB01_TOTAL_FEED_mean'),
            ('CM2_PV_WI01_WATER_INJECTION_mean', 'CM2_PV_VRM01_POWER_mean'),
        ]
        
        for param1, param2 in interactions:
            if all(col in df.columns for col in [param1, param2]):
                base1 = param1.replace('_mean', '').replace('CM2_PV_', '').replace('CM2_SP_', '')
                base2 = param2.replace('_mean', '').replace('CM2_PV_', '').replace('CM2_SP_', '')
                
                # Multiplicative interaction
                df_interact[f'{base1}_x_{base2}'] = df[param1] * df[param2]
                
                # Ratio interaction
                df_interact[f'{base1}_div_{base2}'] = df[param1] / (df[param2] + 1e-6)
                
        self.logger.debug("Created interaction features")
        return df_interact
    
    def create_multi_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features across multiple time scales."""
        df_multi = df.copy()
        
        # Target vibration columns for multi-scale analysis
        vibration_cols = [col for col in df.columns if 'VIBRATION' in col and ('mean' in col or 'std' in col)]
        
        for col in vibration_cols:
            if col in df.columns:
                base_name = col.replace('CM2_PV_VRM01_VIBRATION_', 'vib_')
                
                # Short-term patterns
                for window in self.config.short_windows:
                    df_multi[f'{base_name}_ma_{window}'] = df[col].rolling(window).mean()
                    df_multi[f'{base_name}_std_{window}'] = df[col].rolling(window).std()
                    
                # Medium-term patterns
                for window in self.config.medium_windows:
                    df_multi[f'{base_name}_trend_{window}'] = df[col] - df[col].rolling(window).mean()
                    df_multi[f'{base_name}_volatility_{window}'] = df[col].rolling(window).std()
                    
                # Long-term baseline
                for window in self.config.long_windows:
                    if len(df) > window:
                        df_multi[f'{base_name}_baseline_{window}'] = df[col].rolling(window, min_periods=window//2).mean()
                        df_multi[f'{base_name}_deviation_{window}'] = df[col] - df_multi[f'{base_name}_baseline_{window}']
                        
        self.logger.debug("Created multi-scale features")
        return df_multi
    
    def create_operational_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that identify operational regimes."""
        df_regime = df.copy()
        
        # Power-based regimes
        power_col = self._get_aggregated_cols(df, 'CM2_PV_VRM01_POWER', ['mean'])[0]
        if power_col in df.columns:
            power_q25, power_q75 = df[power_col].quantile([0.25, 0.75])
            df_regime['regime_low_power'] = (df[power_col] <= power_q25).astype(int)
            df_regime['regime_high_power'] = (df[power_col] >= power_q75).astype(int)
            df_regime['regime_normal_power'] = ((df[power_col] > power_q25) & (df[power_col] < power_q75)).astype(int)
            
        # Load-based regimes
        feed_col = self._get_aggregated_cols(df, 'CM2_PV_RB01_TOTAL_FEED', ['mean'])[0]
        if feed_col in df.columns:
            feed_median = df[feed_col].median()
            df_regime['regime_high_load'] = (df[feed_col] > feed_median).astype(int)
            df_regime['regime_low_load'] = (df[feed_col] <= feed_median).astype(int)
            
        # Combined operational state
        if all(col in df_regime.columns for col in ['regime_high_power', 'regime_high_load']):
            df_regime['regime_intensive'] = (df_regime['regime_high_power'] & df_regime['regime_high_load']).astype(int)
            df_regime['regime_light'] = (df_regime['regime_low_power'] & df_regime['regime_low_load']).astype(int)
            
        self.logger.debug("Created operational regime features")
        return df_regime
    
    def _get_aggregated_cols(self, df: pd.DataFrame, base_col: str, suffixes: List[str]) -> List[str]:
        """Helper to find aggregated column names."""
        found_cols = []
        for suffix in suffixes:
            col_name = f'{base_col}_{suffix}'
            if col_name in df.columns:
                found_cols.append(col_name)
        return found_cols
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply all mill-specific feature engineering."""
        self.logger.info("Starting mill-specific feature engineering")
        
        df_features = df.copy()
        
        # Apply feature creation modules
        if self.config.create_efficiency_features:
            df_features = self.create_efficiency_features(df_features)
            
        if self.config.create_stability_features:
            df_features = self.create_stability_features(df_features)
            
        if self.config.create_ratio_features:
            df_features = self.create_process_ratio_features(df_features)
            
        if self.config.create_change_features:
            df_features = self.create_change_detection_features(df_features)
            
        if self.config.create_interaction_features:
            df_features = self.create_interaction_features(df_features)
            
        # Multi-scale features
        df_features = self.create_multi_scale_features(df_features)
        
        # Operational regimes
        df_features = self.create_operational_regime_features(df_features)
        
        # Remove any infinite or extremely large values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        
        # Clip extreme values to reasonable ranges
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            q01, q99 = df_features[col].quantile([0.01, 0.99])
            if q99 > q01:  # Only clip if we have valid range
                df_features[col] = df_features[col].clip(lower=q01, upper=q99)
        
        self.fitted = True
        self.logger.info(f"Mill feature engineering complete: {df.shape[1]} -> {df_features.shape[1]} features")
        
        return df_features
    
    def transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Apply fitted transformations to new data."""
        if not self.fitted:
            self.logger.warning("MillFeatureEngineer not fitted. Using fit_transform instead.")
            return self.fit_transform(df, target_col)
        return self.fit_transform(df, target_col)


def create_mill_feature_config(
    aggressive_features: bool = False,
    target_frequency: str = '5min'
) -> MillFeatureConfig:
    """Create mill feature engineering configuration."""
    if target_frequency == '5min':
        # For 5-minute aggregation
        short = [3, 6, 12]    # 15min, 30min, 1hr
        medium = [24, 48, 96]  # 2hr, 4hr, 8hr  
        long = [144, 288]      # 12hr, 24hr
    elif target_frequency == '15min':
        # For 15-minute aggregation
        short = [2, 4, 8]      # 30min, 1hr, 2hr
        medium = [16, 32, 64]  # 4hr, 8hr, 16hr
        long = [96, 192]       # 24hr, 48hr
    else:
        # Default windows
        short = [3, 6, 12]
        medium = [24, 48]
        long = [96, 144]
        
    return MillFeatureConfig(
        short_windows=short,
        medium_windows=medium,
        long_windows=long,
        create_efficiency_features=True,
        create_stability_features=True,
        create_interaction_features=aggressive_features,
        create_change_features=aggressive_features,
        create_ratio_features=True
    )