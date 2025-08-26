"""
Comprehensive time series evaluation framework with advanced metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, 
    median_absolute_error, explained_variance_score
)
from scipy import stats
from scipy.signal import periodogram
from datetime import datetime, timedelta
import warnings
import logging


class TimeSeriesEvaluator:
    """Comprehensive evaluator for time series prediction models."""
    
    def __init__(self):
        """Initialize the time series evaluator."""
        self.logger = logging.getLogger(__name__)
        
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics."""
        
        # Basic regression metrics
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
        mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        medae = median_absolute_error(y_true, y_pred)
        explained_var = explained_variance_score(y_true, y_pred, sample_weight=sample_weight)
        
        # Additional time series specific metrics
        residuals = y_true - y_pred
        
        # Mean Absolute Percentage Error (MAPE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
            
        # Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        # Mean Absolute Scaled Error (MASE) - comparing to naive forecast
        naive_errors = np.abs(np.diff(y_true)).mean()
        mase = mae / (naive_errors + 1e-8)
        
        # Directional accuracy (for trend prediction)
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
        
        # Prediction interval coverage (assuming normal distribution)
        pred_std = residuals.std()
        coverage_68 = np.mean(np.abs(residuals) <= pred_std) * 100
        coverage_95 = np.mean(np.abs(residuals) <= 1.96 * pred_std) * 100
        
        # Threshold metrics (useful for anomaly detection)
        threshold_1std = np.std(y_true)
        threshold_2std = 2 * threshold_1std
        
        accuracy_1std = np.mean(np.abs(residuals) <= threshold_1std) * 100
        accuracy_2std = np.mean(np.abs(residuals) <= threshold_2std) * 100
        
        return {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'medae': medae,
            'explained_variance': explained_var,
            'mape': mape,
            'smape': smape,
            'mase': mase,
            'directional_accuracy': directional_accuracy,
            'coverage_68': coverage_68,
            'coverage_95': coverage_95,
            'accuracy_1std': accuracy_1std,
            'accuracy_2std': accuracy_2std,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_skew': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals)
        }
    
    def analyze_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """Comprehensive residual analysis."""
        
        residuals = y_true - y_pred
        
        # Statistical tests
        # Normality test
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limit for performance
        
        # Homoscedasticity test (Breusch-Pagan)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            # Add constant column for Breusch-Pagan test
            X_with_const = np.column_stack([np.ones(len(y_pred)), y_pred.reshape(-1, 1)])
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
        except (ImportError, ValueError) as e:
            bp_stat, bp_p = np.nan, np.nan
        
        # Autocorrelation test (Ljung-Box)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(residuals, lags=20, return_df=True)
            lb_stat = lb_result['lb_stat'].iloc[-1]
            lb_p = lb_result['lb_pvalue'].iloc[-1]
        except ImportError:
            lb_stat, lb_p = np.nan, np.nan
        
        # Time-based analysis if timestamps available
        temporal_analysis = {}
        if timestamps is not None:
            # Residual trends over time
            temporal_analysis = self._analyze_temporal_residuals(residuals, timestamps)
        
        return {
            'residuals': residuals,
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals),
            'normality_test': {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'homoscedasticity_test': {
                'bp_stat': bp_stat,
                'bp_p': bp_p,
                'is_homoscedastic': bp_p > 0.05 if not np.isnan(bp_p) else None
            },
            'autocorrelation_test': {
                'ljung_box_stat': lb_stat,
                'ljung_box_p': lb_p,
                'is_uncorrelated': lb_p > 0.05 if not np.isnan(lb_p) else None
            },
            'temporal_analysis': temporal_analysis
        }
    
    def _analyze_temporal_residuals(
        self,
        residuals: np.ndarray,
        timestamps: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """Analyze residuals over time."""
        
        df = pd.DataFrame({
            'residuals': residuals,
            'timestamp': timestamps
        })
        
        # Monthly patterns
        df['month'] = df['timestamp'].dt.month
        monthly_stats = df.groupby('month')['residuals'].agg(['mean', 'std']).to_dict()
        
        # Daily patterns  
        df['hour'] = df['timestamp'].dt.hour
        hourly_stats = df.groupby('hour')['residuals'].agg(['mean', 'std']).to_dict()
        
        # Weekly patterns
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        weekly_stats = df.groupby('dayofweek')['residuals'].agg(['mean', 'std']).to_dict()
        
        # Trend analysis (simple linear regression)
        time_numeric = (timestamps - timestamps[0]).total_seconds()
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, residuals)
        
        return {
            'monthly_patterns': monthly_stats,
            'hourly_patterns': hourly_stats, 
            'weekly_patterns': weekly_stats,
            'trend_analysis': {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_error': std_err,
                'has_trend': p_value < 0.05
            }
        }
    
    def analyze_prediction_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[str, Any]:
        """Analyze prediction intervals and uncertainty."""
        
        residuals = y_true - y_pred
        residual_std = residuals.std()
        
        # Calculate prediction intervals assuming normal distribution
        z_scores = {
            0.68: 1.0,
            0.95: 1.96,
            0.99: 2.58
        }
        
        intervals = {}
        coverage_rates = {}
        
        for confidence in confidence_levels:
            z_score = z_scores.get(confidence, stats.norm.ppf(1 - (1 - confidence) / 2))
            
            # Calculate intervals
            lower_bound = y_pred - z_score * residual_std
            upper_bound = y_pred + z_score * residual_std
            
            # Calculate coverage
            within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            coverage_rate = within_interval.mean()
            
            intervals[confidence] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'width': upper_bound - lower_bound
            }
            
            coverage_rates[confidence] = {
                'actual_coverage': coverage_rate,
                'expected_coverage': confidence,
                'coverage_ratio': coverage_rate / confidence
            }
        
        return {
            'intervals': intervals,
            'coverage_rates': coverage_rates,
            'residual_std': residual_std
        }
    
    def evaluate_forecast_horizons(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: pd.DatetimeIndex,
        horizons: List[str] = ['1H', '4H', '12H', '1D']
    ) -> Dict[str, Any]:
        """Evaluate performance at different forecast horizons."""
        
        if len(timestamps) != len(y_true):
            self.logger.warning("Timestamps length mismatch, skipping horizon analysis")
            return {}
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'true': y_true,
            'pred': y_pred
        }).set_index('timestamp')
        
        horizon_results = {}
        
        for horizon in horizons:
            try:
                # Resample to horizon frequency
                resampled = df.resample(horizon).mean()
                
                if len(resampled) < 10:  # Need enough points
                    continue
                
                # Calculate metrics for this horizon
                metrics = self.calculate_comprehensive_metrics(
                    resampled['true'].values,
                    resampled['pred'].values
                )
                
                horizon_results[horizon] = {
                    'metrics': metrics,
                    'n_points': len(resampled)
                }
                
            except Exception as e:
                self.logger.warning(f"Could not evaluate horizon {horizon}: {e}")
                continue
        
        return horizon_results
    
    def create_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Create a comprehensive evaluation report."""
        
        self.logger.info(f"Creating comprehensive evaluation report for {model_name}")
        
        # Basic metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred)
        
        # Residual analysis
        residual_analysis = self.analyze_residuals(y_true, y_pred, timestamps)
        
        # Prediction intervals
        interval_analysis = self.analyze_prediction_intervals(y_true, y_pred)
        
        # Forecast horizon analysis
        horizon_analysis = {}
        if timestamps is not None:
            horizon_analysis = self.evaluate_forecast_horizons(y_true, y_pred, timestamps)
        
        # Performance categorization
        performance_category = self._categorize_performance(metrics['r2_score'])
        
        report = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'sample_size': len(y_true),
            'performance_category': performance_category,
            'metrics': metrics,
            'residual_analysis': residual_analysis,
            'interval_analysis': interval_analysis,
            'horizon_analysis': horizon_analysis,
            'summary': self._create_summary_text(metrics, residual_analysis)
        }
        
        return report
    
    def _categorize_performance(self, r2_score: float) -> Dict[str, Any]:
        """Categorize model performance."""
        
        if r2_score >= 0.8:
            category = "Excellent"
            color = "green"
            description = "Model explains >80% of variance with high predictive accuracy"
        elif r2_score >= 0.6:
            category = "Good"
            color = "lightgreen"
            description = "Model shows good predictive performance with room for minor improvements"
        elif r2_score >= 0.4:
            category = "Moderate"
            color = "yellow"
            description = "Model shows moderate predictive ability but needs improvement"
        elif r2_score >= 0.2:
            category = "Fair"
            color = "orange"
            description = "Model shows some predictive ability but significant room for improvement"
        elif r2_score >= 0:
            category = "Poor"
            color = "red"
            description = "Model performs poorly, barely better than mean prediction"
        else:
            category = "Very Poor"
            color = "darkred"
            description = "Model performs worse than simply predicting the mean"
        
        return {
            'category': category,
            'color': color,
            'description': description,
            'r2_score': r2_score
        }
    
    def _create_summary_text(
        self,
        metrics: Dict[str, float],
        residual_analysis: Dict[str, Any]
    ) -> str:
        """Create a text summary of the evaluation."""
        
        summary_lines = [
            f"Model Performance Summary:",
            f"- R² Score: {metrics['r2_score']:.4f}",
            f"- RMSE: {metrics['rmse']:.4f}",
            f"- MAE: {metrics['mae']:.4f}",
            f"- MAPE: {metrics['mape']:.2f}%",
            f"- Directional Accuracy: {metrics['directional_accuracy']:.1f}%",
            "",
            "Residual Analysis:",
            f"- Mean: {residual_analysis['mean']:.4f}",
            f"- Std Dev: {residual_analysis['std']:.4f}",
            f"- Skewness: {residual_analysis['skewness']:.3f}",
        ]
        
        # Add normality test result
        if residual_analysis['normality_test']['is_normal']:
            summary_lines.append("- Residuals appear normally distributed ✓")
        else:
            summary_lines.append("- Residuals may not be normally distributed ⚠")
        
        # Add autocorrelation test result
        autocorr = residual_analysis['autocorrelation_test']
        if autocorr['is_uncorrelated'] is not None:
            if autocorr['is_uncorrelated']:
                summary_lines.append("- No significant autocorrelation detected ✓")
            else:
                summary_lines.append("- Autocorrelation detected in residuals ⚠")
        
        return "\n".join(summary_lines)
    
    def create_evaluation_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (20, 15)
    ) -> plt.Figure:
        """Create comprehensive evaluation plots."""
        
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        fig.suptitle(f'Comprehensive Model Evaluation: {model_name}', fontsize=16, y=0.98)
        
        residuals = y_true - y_pred
        
        # Plot 1: Predictions vs Actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=1)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        r2 = r2_score(y_true, y_pred)
        axes[0, 0].set_title(f'Predictions vs Actual\nR² = {r2:.4f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals vs Predicted
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=1)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residual Distribution
        axes[0, 2].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(residuals.mean(), color='r', linestyle='--', 
                          label=f'Mean: {residuals.mean():.4f}')
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Residual Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 3])
        axes[0, 3].set_title('Q-Q Plot (Normality Check)')
        axes[0, 3].grid(True, alpha=0.3)
        
        # Plot 5: Time Series (if timestamps available)
        if timestamps is not None:
            sample_size = min(2000, len(y_true))
            sample_idx = np.random.choice(len(y_true), sample_size, replace=False)
            sample_idx = np.sort(sample_idx)
            
            axes[1, 0].plot(timestamps[sample_idx], y_true[sample_idx], 
                           label='Actual', alpha=0.7, linewidth=1)
            axes[1, 0].plot(timestamps[sample_idx], y_pred[sample_idx], 
                           label='Predicted', alpha=0.7, linewidth=1)
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Time Series Comparison (Sample)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 6: Residuals over time
            axes[1, 1].plot(timestamps[sample_idx], residuals[sample_idx], 
                           alpha=0.7, linewidth=1)
            axes[1, 1].axhline(y=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals Over Time')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Time series plot\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Time Series Comparison')
            
            axes[1, 1].text(0.5, 0.5, 'Temporal residuals\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Residuals Over Time')
        
        # Plot 7: Error Distribution by Magnitude
        abs_residuals = np.abs(residuals)
        percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
        perc_values = np.percentile(abs_residuals, percentiles)
        
        axes[1, 2].bar(range(len(percentiles)), perc_values, alpha=0.7)
        axes[1, 2].set_xticks(range(len(percentiles)))
        axes[1, 2].set_xticklabels([f'{p}%' for p in percentiles])
        axes[1, 2].set_xlabel('Percentile')
        axes[1, 2].set_ylabel('Absolute Error')
        axes[1, 2].set_title('Error Distribution by Percentile')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Plot 8: Prediction Intervals
        interval_analysis = self.analyze_prediction_intervals(y_true, y_pred)
        coverage_68 = interval_analysis['coverage_rates'][0.68]['actual_coverage']
        coverage_95 = interval_analysis['coverage_rates'][0.95]['actual_coverage']
        
        sample_size = min(500, len(y_true))
        sample_idx = np.random.choice(len(y_true), sample_size, replace=False)
        sample_idx = np.sort(sample_idx)
        
        x_axis = range(len(sample_idx))
        lower_68 = interval_analysis['intervals'][0.68]['lower_bound'][sample_idx]
        upper_68 = interval_analysis['intervals'][0.68]['upper_bound'][sample_idx]
        lower_95 = interval_analysis['intervals'][0.95]['lower_bound'][sample_idx]
        upper_95 = interval_analysis['intervals'][0.95]['upper_bound'][sample_idx]
        
        axes[1, 3].fill_between(x_axis, lower_95, upper_95, alpha=0.2, color='blue', label='95% PI')
        axes[1, 3].fill_between(x_axis, lower_68, upper_68, alpha=0.3, color='blue', label='68% PI')
        axes[1, 3].scatter(x_axis, y_true[sample_idx], s=1, alpha=0.6, color='red', label='Actual')
        axes[1, 3].plot(x_axis, y_pred[sample_idx], color='black', linewidth=1, label='Predicted')
        
        axes[1, 3].set_xlabel('Sample Index')
        axes[1, 3].set_ylabel('Value')
        axes[1, 3].set_title(f'Prediction Intervals\n68%: {coverage_68:.1f}%, 95%: {coverage_95:.1f}%')
        axes[1, 3].legend(fontsize=8)
        axes[1, 3].grid(True, alpha=0.3)
        
        # Bottom row: Performance metrics and summary
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred)
        
        # Plot 9: Metrics Bar Chart
        metric_names = ['R²', 'MAPE', 'Dir. Acc.', 'Coverage 68%', 'Coverage 95%']
        metric_values = [
            metrics['r2_score'] * 100,  # Convert to percentage
            min(metrics['mape'], 100),   # Cap MAPE at 100%
            metrics['directional_accuracy'],
            metrics['coverage_68'],
            metrics['coverage_95']
        ]
        
        colors = ['green' if v > 80 else 'orange' if v > 60 else 'red' for v in metric_values]
        bars = axes[2, 0].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[2, 0].set_ylabel('Percentage')
        axes[2, 0].set_title('Key Performance Metrics')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[2, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 10: Error by Value Range
        try:
            # Bin actual values and calculate error statistics
            n_bins = 10
            bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            bin_indices = np.digitize(y_true, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            bin_errors = []
            for i in range(n_bins):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    bin_errors.append(np.mean(np.abs(residuals[mask])))
                else:
                    bin_errors.append(0)
            
            axes[2, 1].bar(bin_centers, bin_errors, width=np.diff(bins)[0] * 0.8, alpha=0.7)
            axes[2, 1].set_xlabel('Actual Value Range')
            axes[2, 1].set_ylabel('Mean Absolute Error')
            axes[2, 1].set_title('Error by Value Range')
            axes[2, 1].grid(True, alpha=0.3)
        except Exception:
            axes[2, 1].text(0.5, 0.5, 'Error analysis\nnot available', 
                           ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Error by Value Range')
        
        # Plot 11: Autocorrelation of Residuals
        try:
            from statsmodels.tsa.stattools import acf
            autocorr = acf(residuals, nlags=20, fft=True)
            lags = range(len(autocorr))
            axes[2, 2].plot(lags, autocorr, 'o-', markersize=3)
            axes[2, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2, 2].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
            axes[2, 2].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
            axes[2, 2].set_xlabel('Lag')
            axes[2, 2].set_ylabel('Autocorrelation')
            axes[2, 2].set_title('Residual Autocorrelation')
            axes[2, 2].grid(True, alpha=0.3)
        except ImportError:
            axes[2, 2].text(0.5, 0.5, 'Autocorrelation plot\nrequires statsmodels', 
                           ha='center', va='center', transform=axes[2, 2].transAxes)
            axes[2, 2].set_title('Residual Autocorrelation')
        
        # Plot 12: Summary Text
        axes[2, 3].axis('off')
        
        # Performance category
        performance = self._categorize_performance(metrics['r2_score'])
        
        summary_text = [
            f"MODEL: {model_name}",
            f"PERFORMANCE: {performance['category']}",
            "",
            f"R² Score: {metrics['r2_score']:.4f}",
            f"RMSE: {metrics['rmse']:.4f}",
            f"MAE: {metrics['mae']:.4f}",
            f"MAPE: {metrics['mape']:.2f}%",
            f"SMAPE: {metrics['smape']:.2f}%",
            f"Direction Acc: {metrics['directional_accuracy']:.1f}%",
            "",
            f"Samples: {len(y_true):,}",
            f"Mean Residual: {residuals.mean():.4f}",
            f"Residual Std: {residuals.std():.4f}",
            "",
            performance['description']
        ]
        
        axes[2, 3].text(0.05, 0.95, '\n'.join(summary_text), 
                       transform=axes[2, 3].transAxes, fontsize=10,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor=performance['color'], alpha=0.3))
        
        plt.tight_layout()
        return fig


def create_time_series_evaluator() -> TimeSeriesEvaluator:
    """Create a time series evaluator instance."""
    return TimeSeriesEvaluator()