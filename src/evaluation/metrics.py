"""
Model evaluation metrics and utilities for the roller mill vibration prediction project.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    mean_absolute_percentage_error, explained_variance_score
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

from config.settings import get_config


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model evaluator.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def calculate_regression_metrics(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of regression metrics.
        """
        # Convert to numpy arrays and handle NaN values
        y_true_clean, y_pred_clean = self._clean_arrays(y_true, y_pred)
        
        if len(y_true_clean) == 0:
            self.logger.warning("No valid predictions for metric calculation")
            return {}
        
        metrics = {}
        
        try:
            # Basic regression metrics
            metrics['r2_score'] = r2_score(y_true_clean, y_pred_clean)
            metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
            
            # Mean Absolute Percentage Error (only if no zeros in y_true)
            if not np.any(y_true_clean == 0):
                metrics['mape'] = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
            
            # Explained variance score
            metrics['explained_variance'] = explained_variance_score(y_true_clean, y_pred_clean)
            
            # Median Absolute Error
            metrics['median_ae'] = np.median(np.abs(y_true_clean - y_pred_clean))
            
            # Maximum error
            metrics['max_error'] = np.max(np.abs(y_true_clean - y_pred_clean))
            
            # Correlation metrics
            pearson_corr, pearson_p = pearsonr(y_true_clean, y_pred_clean)
            spearman_corr, spearman_p = spearmanr(y_true_clean, y_pred_clean)
            
            metrics['pearson_correlation'] = pearson_corr
            metrics['pearson_p_value'] = pearson_p
            metrics['spearman_correlation'] = spearman_corr
            metrics['spearman_p_value'] = spearman_p
            
            # Normalized metrics
            y_range = np.max(y_true_clean) - np.min(y_true_clean)
            if y_range > 0:
                metrics['nrmse'] = metrics['rmse'] / y_range  # Normalized RMSE
                metrics['nmae'] = metrics['mae'] / y_range    # Normalized MAE
            
            # Bias and direction accuracy
            residuals = y_true_clean - y_pred_clean
            metrics['mean_bias'] = np.mean(residuals)
            metrics['bias_variance'] = np.var(residuals)
            
            # Direction accuracy (for time series)
            if len(y_true_clean) > 1:
                true_directions = np.diff(y_true_clean) > 0
                pred_directions = np.diff(y_pred_clean) > 0
                metrics['direction_accuracy'] = np.mean(true_directions == pred_directions)
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def calculate_time_series_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        time_index: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, float]:
        """Calculate time series specific metrics.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            time_index: Time index for the series.
            
        Returns:
            Dictionary of time series metrics.
        """
        y_true_clean, y_pred_clean = self._clean_arrays(y_true, y_pred)
        
        if len(y_true_clean) < 2:
            return {}
        
        metrics = {}
        
        try:
            # Trend accuracy
            true_trend = np.polyfit(range(len(y_true_clean)), y_true_clean, 1)[0]
            pred_trend = np.polyfit(range(len(y_pred_clean)), y_pred_clean, 1)[0]
            metrics['trend_accuracy'] = 1 - abs(true_trend - pred_trend) / (abs(true_trend) + 1e-8)
            
            # Seasonality metrics (if time index provided)
            if time_index is not None and len(time_index) == len(y_true_clean):
                try:
                    # Daily seasonality (if data spans multiple days)
                    if (time_index.max() - time_index.min()).days > 1:
                        true_hourly = y_true_clean.groupby(time_index.hour).mean()
                        pred_hourly = y_pred_clean.groupby(time_index.hour).mean()
                        if len(true_hourly) > 1:
                            hourly_corr, _ = pearsonr(true_hourly, pred_hourly)
                            metrics['hourly_pattern_correlation'] = hourly_corr
                    
                    # Weekly seasonality (if data spans multiple weeks)
                    if (time_index.max() - time_index.min()).days > 7:
                        true_daily = y_true_clean.groupby(time_index.dayofweek).mean()
                        pred_daily = y_pred_clean.groupby(time_index.dayofweek).mean()
                        if len(true_daily) > 1:
                            daily_corr, _ = pearsonr(true_daily, pred_daily)
                            metrics['daily_pattern_correlation'] = daily_corr
                            
                except Exception as e:
                    self.logger.warning(f"Error calculating seasonality metrics: {e}")
            
            # Persistence model comparison (naive forecast)
            if len(y_true_clean) > 1:
                naive_pred = np.roll(y_true_clean, 1)[1:]  # Shift by one
                true_subset = y_true_clean[1:]
                pred_subset = y_pred_clean[1:]
                
                naive_mse = mean_squared_error(true_subset, naive_pred)
                model_mse = mean_squared_error(true_subset, pred_subset)
                
                # Skill score (improvement over naive model)
                metrics['skill_score'] = 1 - (model_mse / naive_mse) if naive_mse > 0 else 0
            
            # Forecast error statistics
            errors = y_true_clean - y_pred_clean
            
            # Mean Absolute Scaled Error (MASE)
            if len(y_true_clean) > 1:
                naive_mae = np.mean(np.abs(np.diff(y_true_clean)))
                if naive_mae > 0:
                    metrics['mase'] = np.mean(np.abs(errors)) / naive_mae
            
            # Autocorrelation of residuals (should be low for good models)
            if len(errors) > 10:
                from statsmodels.tsa.stattools import acf
                try:
                    autocorr = acf(errors, nlags=1, fft=False)[1]
                    metrics['residual_autocorrelation'] = autocorr
                except Exception:
                    pass  # Skip if statsmodels not available or fails
            
        except Exception as e:
            self.logger.error(f"Error calculating time series metrics: {e}")
        
        return metrics
    
    def evaluate_model_residuals(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze model residuals for diagnostic purposes.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Dictionary of residual analysis results.
        """
        y_true_clean, y_pred_clean = self._clean_arrays(y_true, y_pred)
        
        if len(y_true_clean) == 0:
            return {}
        
        residuals = y_true_clean - y_pred_clean
        
        analysis = {
            'residual_stats': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'median': np.median(residuals),
                'q25': np.percentile(residuals, 25),
                'q75': np.percentile(residuals, 75)
            },
            'normality_test': {},
            'homoscedasticity_test': {},
            'outliers': {}
        }
        
        try:
            # Normality test (Shapiro-Wilk for smaller samples, Anderson-Darling for larger)
            if len(residuals) <= 5000:
                stat, p_value = stats.shapiro(residuals)
                analysis['normality_test'] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
            else:
                # Use Anderson-Darling test for larger samples
                result = stats.anderson(residuals, dist='norm')
                analysis['normality_test'] = {
                    'test': 'Anderson-Darling',
                    'statistic': result.statistic,
                    'critical_values': result.critical_values.tolist(),
                    'significance_levels': result.significance_levels.tolist()
                }
            
            # Homoscedasticity (constant variance) test using Breusch-Pagan
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                
                # Simple linear regression of residuals squared on predictions
                X = np.column_stack([np.ones(len(y_pred_clean)), y_pred_clean])
                lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, X)
                
                analysis['homoscedasticity_test'] = {
                    'test': 'Breusch-Pagan',
                    'lm_statistic': lm_stat,
                    'lm_p_value': lm_pvalue,
                    'f_statistic': f_stat,
                    'f_p_value': f_pvalue,
                    'is_homoscedastic': lm_pvalue > 0.05
                }
            except ImportError:
                # Fallback to simple variance ratio test
                mid_point = len(residuals) // 2
                first_half_var = np.var(residuals[:mid_point])
                second_half_var = np.var(residuals[mid_point:])
                
                analysis['homoscedasticity_test'] = {
                    'test': 'Variance Ratio',
                    'ratio': second_half_var / (first_half_var + 1e-8),
                    'note': 'Simple variance comparison between first and second half'
                }
            
            # Outlier detection using IQR method
            q1 = np.percentile(residuals, 25)
            q3 = np.percentile(residuals, 75)
            iqr = q3 - q1
            
            outlier_threshold_low = q1 - 1.5 * iqr
            outlier_threshold_high = q3 + 1.5 * iqr
            
            outliers_mask = (residuals < outlier_threshold_low) | (residuals > outlier_threshold_high)
            outlier_indices = np.where(outliers_mask)[0]
            
            analysis['outliers'] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(residuals)) * 100,
                'indices': outlier_indices.tolist(),
                'threshold_low': outlier_threshold_low,
                'threshold_high': outlier_threshold_high
            }
            
        except Exception as e:
            self.logger.error(f"Error in residual analysis: {e}")
        
        return analysis
    
    def generate_evaluation_plots(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        model_name: str = "Model",
        time_index: Optional[pd.DatetimeIndex] = None,
        save_path: Optional[str] = None
    ) -> None:
        """Generate comprehensive evaluation plots.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            model_name: Name of the model for plot titles.
            time_index: Time index for time series plots.
            save_path: Directory to save plots.
        """
        y_true_clean, y_pred_clean = self._clean_arrays(y_true, y_pred)
        
        if len(y_true_clean) == 0:
            self.logger.warning("No valid data for plotting")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Predictions vs Actual scatter plot
        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(y_true_clean, y_pred_clean, alpha=0.6, s=1)
        
        # Perfect prediction line
        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Predictions vs Actual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score to the plot
        r2 = r2_score(y_true_clean, y_pred_clean)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Residuals plot
        ax2 = plt.subplot(3, 3, 2)
        residuals = y_true_clean - y_pred_clean
        plt.scatter(y_pred_clean, residuals, alpha=0.6, s=1)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title(f'{model_name}: Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        ax3 = plt.subplot(3, 3, 3)
        plt.hist(residuals, bins=50, alpha=0.7, density=True, color='skyblue')
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, normal_curve, 'r-', lw=2, label='Normal Distribution')
        
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title(f'{model_name}: Residuals Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Q-Q plot for residuals
        ax4 = plt.subplot(3, 3, 4)
        stats.probplot(residuals, dist="norm", plot=ax4)
        plt.title(f'{model_name}: Q-Q Plot (Residuals)')
        plt.grid(True, alpha=0.3)
        
        # 5. Time series plot (if time index provided)
        if time_index is not None and len(time_index) == len(y_true_clean):
            ax5 = plt.subplot(3, 3, 5)
            
            # Sample data if too many points
            if len(time_index) > 5000:
                sample_indices = np.random.choice(len(time_index), 5000, replace=False)
                sample_indices = np.sort(sample_indices)
                time_sample = time_index[sample_indices]
                true_sample = y_true_clean[sample_indices]
                pred_sample = y_pred_clean[sample_indices]
            else:
                time_sample = time_index
                true_sample = y_true_clean
                pred_sample = y_pred_clean
            
            plt.plot(time_sample, true_sample, label='Actual', alpha=0.7, linewidth=1)
            plt.plot(time_sample, pred_sample, label='Predicted', alpha=0.7, linewidth=1)
            plt.xlabel('Time')
            plt.ylabel('Values')
            plt.title(f'{model_name}: Time Series Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # 6. Error distribution over time (if time index provided)
        if time_index is not None and len(time_index) == len(y_true_clean):
            ax6 = plt.subplot(3, 3, 6)
            
            # Use same sampling as above
            if len(time_index) > 5000:
                error_sample = residuals[sample_indices]
            else:
                error_sample = residuals
                time_sample = time_index
            
            plt.scatter(time_sample, error_sample, alpha=0.6, s=1)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Time')
            plt.ylabel('Prediction Error')
            plt.title(f'{model_name}: Error Over Time')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        # 7. Absolute error distribution
        ax7 = plt.subplot(3, 3, 7)
        abs_errors = np.abs(residuals)
        plt.hist(abs_errors, bins=50, alpha=0.7, color='lightcoral')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Absolute Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mae = np.mean(abs_errors)
        plt.axvline(x=mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.3f}')
        plt.legend()
        
        # 8. Error vs predicted value (different view of residuals)
        ax8 = plt.subplot(3, 3, 8)
        plt.scatter(y_pred_clean, np.abs(residuals), alpha=0.6, s=1, color='orange')
        plt.xlabel('Predicted Values')
        plt.ylabel('Absolute Error')
        plt.title(f'{model_name}: Absolute Error vs Predictions')
        plt.grid(True, alpha=0.3)
        
        # 9. Model performance metrics summary
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate metrics
        metrics = self.calculate_regression_metrics(y_true_clean, y_pred_clean)
        
        # Create text summary
        metrics_text = f"""Model Performance Metrics:
        
R² Score: {metrics.get('r2_score', 0):.4f}
RMSE: {metrics.get('rmse', 0):.4f}
MAE: {metrics.get('mae', 0):.4f}
MAPE: {metrics.get('mape', 0):.2f}%
Max Error: {metrics.get('max_error', 0):.4f}

Correlation:
Pearson: {metrics.get('pearson_correlation', 0):.4f}
Spearman: {metrics.get('spearman_correlation', 0):.4f}

Sample Size: {len(y_true_clean):,}
        """
        
        ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plot_file = save_path / f"{model_name.replace(' ', '_')}_evaluation.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            self.logger.info(f"Evaluation plots saved to {plot_file}")
        
        plt.show()
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, Any]],
        metrics_to_compare: List[str] = None
    ) -> pd.DataFrame:
        """Compare multiple models based on evaluation metrics.
        
        Args:
            results: Dictionary mapping model names to their evaluation results.
            metrics_to_compare: List of metrics to include in comparison.
            
        Returns:
            DataFrame with model comparison.
        """
        if metrics_to_compare is None:
            metrics_to_compare = ['r2_score', 'rmse', 'mae', 'mape', 'pearson_correlation']
        
        comparison_data = []
        
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            
            for metric in metrics_to_compare:
                value = model_results.get(metric, np.nan)
                row[metric] = value
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by R² score (descending) if available
        if 'r2_score' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('r2_score', ascending=False)
        
        # Add ranking
        comparison_df.reset_index(drop=True, inplace=True)
        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Model'] + [col for col in comparison_df.columns if col not in ['Rank', 'Model']]
        comparison_df = comparison_df[cols]
        
        return comparison_df
    
    def _clean_arrays(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: Union[pd.Series, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clean arrays by removing NaN and infinite values.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
            
        Returns:
            Tuple of cleaned arrays.
        """
        # Convert to numpy arrays
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        # Find valid (non-NaN, non-infinite) indices
        valid_mask = (
            np.isfinite(y_true_arr) & 
            np.isfinite(y_pred_arr) & 
            ~np.isnan(y_true_arr) & 
            ~np.isnan(y_pred_arr)
        )
        
        if np.sum(valid_mask) < len(y_true_arr):
            removed_count = len(y_true_arr) - np.sum(valid_mask)
            self.logger.warning(f"Removed {removed_count} invalid predictions for evaluation")
        
        return y_true_arr[valid_mask], y_pred_arr[valid_mask]
    
    def save_evaluation_report(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        residual_analysis: Dict[str, Any],
        output_path: Union[str, Path]
    ) -> None:
        """Save a comprehensive evaluation report.
        
        Args:
            model_name: Name of the model.
            metrics: Evaluation metrics.
            residual_analysis: Residual analysis results.
            output_path: Path to save the report.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create markdown report
        report = f"""# Model Evaluation Report: {model_name}

## Model Performance Metrics

### Regression Metrics
"""
        
        # Add regression metrics
        if metrics:
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    report += f"- **{metric}**: {value:.4f}\n"
                else:
                    report += f"- **{metric}**: {value}\n"
        
        # Add residual analysis
        if residual_analysis:
            report += "\n## Residual Analysis\n\n"
            
            if 'residual_stats' in residual_analysis:
                report += "### Residual Statistics\n"
                stats = residual_analysis['residual_stats']
                for stat, value in stats.items():
                    report += f"- **{stat}**: {value:.4f}\n"
            
            if 'normality_test' in residual_analysis:
                report += "\n### Normality Test\n"
                test = residual_analysis['normality_test']
                report += f"- **Test**: {test.get('test', 'Unknown')}\n"
                if 'p_value' in test:
                    report += f"- **P-value**: {test['p_value']:.6f}\n"
                    report += f"- **Is Normal**: {'Yes' if test.get('is_normal', False) else 'No'}\n"
            
            if 'outliers' in residual_analysis:
                report += "\n### Outliers\n"
                outliers = residual_analysis['outliers']
                report += f"- **Count**: {outliers.get('count', 0)}\n"
                report += f"- **Percentage**: {outliers.get('percentage', 0):.2f}%\n"
        
        # Add timestamp
        from datetime import datetime
        report += f"\n---\n*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Evaluation report saved to {output_path}")


def evaluate_model_comprehensive(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: Optional[str] = None,
    time_index: Optional[pd.DatetimeIndex] = None,
    save_plots: bool = True,
    save_report: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Perform comprehensive model evaluation.
    
    Args:
        model: Trained model with predict method.
        X_test: Test features.
        y_test: Test target.
        model_name: Name of the model.
        time_index: Time index for time series analysis.
        save_plots: Whether to save evaluation plots.
        save_report: Whether to save evaluation report.
        output_dir: Directory to save outputs.
        
    Returns:
        Complete evaluation results.
    """
    if model_name is None:
        model_name = getattr(model, 'name', 'Unknown Model')
    
    if output_dir is None:
        config = get_config()
        output_dir = config.paths.plots
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    regression_metrics = evaluator.calculate_regression_metrics(y_test, y_pred)
    
    # Time series metrics
    ts_metrics = {}
    if time_index is not None:
        ts_metrics = evaluator.calculate_time_series_metrics(y_test, y_pred, time_index)
    
    # Residual analysis
    residual_analysis = evaluator.evaluate_model_residuals(y_test, y_pred)
    
    # Generate plots
    if save_plots:
        evaluator.generate_evaluation_plots(
            y_test, y_pred, model_name, time_index, output_dir
        )
    
    # Save report
    if save_report:
        report_path = Path(output_dir) / f"{model_name.replace(' ', '_')}_evaluation_report.md"
        evaluator.save_evaluation_report(
            model_name, regression_metrics, residual_analysis, report_path
        )
    
    # Compile results
    results = {
        'model_name': model_name,
        'predictions': y_pred,
        'regression_metrics': regression_metrics,
        'time_series_metrics': ts_metrics,
        'residual_analysis': residual_analysis,
        'sample_size': len(y_test)
    }
    
    return results