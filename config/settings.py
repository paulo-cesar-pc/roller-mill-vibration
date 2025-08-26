"""
Configuration management for the roller mill vibration prediction project.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class TestingConfig:
    """Testing configuration settings."""
    enabled: bool
    max_rows: int


@dataclass
class DataConfig:
    """Data configuration settings."""
    raw_data_path: str
    processed_data_path: str
    target_column: str
    timestamp_column: str
    timestamp_format: str
    filter_vibration_range: Dict[str, float]
    train_split: float
    validation_split: float
    test_split: float
    testing: TestingConfig


@dataclass
class FeatureConfig:
    """Feature engineering configuration settings."""
    high_corr_features: List[str]
    categorical_columns: List[str]
    lag_features: Dict[str, Any]
    rolling_features: Dict[str, Any]
    temporal_features: Dict[str, Any]
    spectral_features: Dict[str, Any]
    seasonality_features: Dict[str, Any]


@dataclass
class ModelConfig:
    """Model configuration settings."""
    linear_regression: Dict[str, Any]
    random_forest: Dict[str, Any]
    xgboost: Dict[str, Any]
    lstm: Dict[str, Any]
    ensemble: Dict[str, Any]


@dataclass
class TrainingConfig:
    """Training configuration settings."""
    cv_folds: int
    cv_method: str
    optimization: Dict[str, Any]
    selection_metric: str
    early_stopping: Dict[str, Any]


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings."""
    metrics: List[str]
    plots: List[str]


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str
    format: str
    file: str


@dataclass
class MLflowConfig:
    """MLflow configuration settings."""
    enabled: bool
    tracking_uri: str
    experiment_name: str


@dataclass
class PathsConfig:
    """Paths configuration settings."""
    data_raw: str
    data_processed: str
    models: str
    logs: str
    plots: str
    experiments: str


@dataclass
class ProjectConfig:
    """Main project configuration."""
    name: str
    version: str
    description: str


@dataclass
class Config:
    """Complete configuration object."""
    project: ProjectConfig
    data: DataConfig
    features: FeatureConfig
    models: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    mlflow: MLflowConfig
    paths: PathsConfig


class ConfigManager:
    """Configuration manager for loading and managing configuration settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            self.config_path = Path(__file__).parent / "config.yaml"
        else:
            self.config_path = Path(config_path)
            
        self.config = self._load_config()
        self._setup_directories()
    
    def _load_config(self) -> Config:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_dict = yaml.safe_load(file)
            
            # Create configuration objects
            project_config = ProjectConfig(**config_dict['project'])
            
            # Handle nested testing config within data config
            data_dict = config_dict['data'].copy()
            testing_dict = data_dict.pop('testing', {'enabled': False, 'max_rows': 10000})
            testing_config = TestingConfig(**testing_dict)
            data_config = DataConfig(**data_dict, testing=testing_config)
            features_config = FeatureConfig(**config_dict['features'])
            models_config = ModelConfig(**config_dict['models'])
            training_config = TrainingConfig(**config_dict['training'])
            evaluation_config = EvaluationConfig(**config_dict['evaluation'])
            logging_config = LoggingConfig(**config_dict['logging'])
            mlflow_config = MLflowConfig(**config_dict['mlflow'])
            paths_config = PathsConfig(**config_dict['paths'])
            
            return Config(
                project=project_config,
                data=data_config,
                features=features_config,
                models=models_config,
                training=training_config,
                evaluation=evaluation_config,
                logging=logging_config,
                mlflow=mlflow_config,
                paths=paths_config
            )
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.config.paths.data_raw,
            self.config.paths.data_processed,
            self.config.paths.models,
            self.config.paths.logs,
            self.config.paths.plots,
            self.config.paths.experiments
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get parameters for a specific model.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            Dictionary of model parameters.
        """
        model_configs = {
            'linear_regression': self.config.models.linear_regression,
            'random_forest': self.config.models.random_forest,
            'xgboost': self.config.models.xgboost,
            'lstm': self.config.models.lstm,
            'ensemble': self.config.models.ensemble
        }
        
        if model_name not in model_configs:
            raise ValueError(f"Model {model_name} not found in configuration")
        
        return model_configs[model_name]
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns to use."""
        return self.config.features.high_corr_features + self.config.features.categorical_columns
    
    def is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled in the configuration.
        
        Args:
            model_name: Name of the model.
            
        Returns:
            True if model is enabled, False otherwise.
        """
        model_config = self.get_model_params(model_name)
        return model_config.get('enabled', False)
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update a configuration value.
        
        Args:
            section: Configuration section (e.g., 'data', 'models').
            key: Configuration key to update.
            value: New value.
        """
        if hasattr(self.config, section):
            section_obj = getattr(self.config, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
            else:
                raise KeyError(f"Key '{key}' not found in section '{section}'")
        else:
            raise KeyError(f"Section '{section}' not found in configuration")
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """Save current configuration to YAML file.
        
        Args:
            output_path: Path to save the configuration. If None, overwrites current file.
        """
        if output_path is None:
            output_path = self.config_path
        
        # Convert config objects back to dictionaries
        config_dict = {
            'project': {
                'name': self.config.project.name,
                'version': self.config.project.version,
                'description': self.config.project.description
            },
            'data': {
                'raw_data_path': self.config.data.raw_data_path,
                'processed_data_path': self.config.data.processed_data_path,
                'target_column': self.config.data.target_column,
                'timestamp_column': self.config.data.timestamp_column,
                'timestamp_format': self.config.data.timestamp_format,
                'filter_vibration_range': self.config.data.filter_vibration_range,
                'train_split': self.config.data.train_split,
                'validation_split': self.config.data.validation_split,
                'test_split': self.config.data.test_split
            },
            # Add other sections as needed
        }
        
        with open(output_path, 'w', encoding='utf-8') as file:
            yaml.dump(config_dict, file, default_flow_style=False, indent=2)


# Singleton configuration manager
_config_manager = None


def get_config() -> Config:
    """Get the global configuration object.
    
    Returns:
        Configuration object.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager.
    
    Returns:
        Configuration manager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


# Environment-specific configurations
def get_config_for_environment(env: str = "development") -> Config:
    """Get configuration for a specific environment.
    
    Args:
        env: Environment name (development, staging, production).
        
    Returns:
        Configuration object for the specified environment.
    """
    env_config_path = Path(__file__).parent / f"config_{env}.yaml"
    
    if env_config_path.exists():
        manager = ConfigManager(str(env_config_path))
        return manager.config
    else:
        # Fall back to default configuration
        return get_config()