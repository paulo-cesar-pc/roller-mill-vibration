# Roller Mill Vibration Prediction

A comprehensive machine learning project for predicting roller mill vibration using advanced time series modeling techniques.

## Overview

This project transforms a basic Jupyter notebook approach into a production-ready ML system for predicting roller mill vibration. It includes advanced feature engineering, multiple model architectures, ensemble methods, and comprehensive evaluation metrics.

## Key Improvements from Original Notebook

- ✅ **Proper Project Structure**: Modular, maintainable code with clear separation of concerns
- ✅ **Advanced Feature Engineering**: Spectral analysis, seasonality decomposition, lag features, rolling statistics
- ✅ **Multiple Model Types**: Traditional ML (XGBoost, Random Forest) + Deep Learning (LSTM, GRU)
- ✅ **Automated Hyperparameter Optimization**: Using Optuna for efficient parameter search
- ✅ **Ensemble Methods**: Model stacking and voting for improved performance
- ✅ **Comprehensive Evaluation**: Time series specific metrics, residual analysis, statistical tests
- ✅ **Data Validation**: Automated data quality checks and validation pipelines
- ✅ **Experiment Tracking**: MLflow integration for reproducible experiments
- ✅ **Configuration Management**: Centralized, flexible configuration system

## Project Structure

```
roller-mill-vibration/
├── config/                 # Configuration files
│   ├── config.yaml         # Main configuration
│   └── settings.py         # Configuration management
├── data/                   # Data storage
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data
│   └── external/          # External data sources
├── src/                   # Source code
│   ├── data/              # Data processing
│   │   ├── data_loader.py # Data loading and preprocessing
│   │   └── validators.py  # Data validation
│   ├── features/          # Feature engineering
│   │   └── feature_engineer.py
│   ├── models/            # Model implementations
│   │   ├── base_model.py  # Base model classes
│   │   ├── lstm_model.py  # Deep learning models
│   │   └── trainer.py     # Training pipeline
│   ├── evaluation/        # Model evaluation
│   │   └── metrics.py     # Evaluation metrics
│   └── utils/             # Utility functions
├── experiments/           # Experiment results
├── models/               # Trained models
├── notebooks/            # Research notebooks
├── tests/                # Unit tests
├── scripts/              # Training scripts
│   └── train_models.py   # Main training script
├── requirements.txt      # Python dependencies
├── pyproject.toml       # Project configuration
└── README.md           # This file
```

## Features

### Advanced Feature Engineering
- **Temporal Features**: Cyclical encoding of time components (hour, day, week)
- **Lag Features**: Multiple lag values for capturing temporal dependencies
- **Rolling Statistics**: Mean, std, min, max over multiple time windows
- **Spectral Features**: FFT-based frequency domain analysis
- **Seasonality Features**: Trend and seasonal decomposition
- **Interaction Features**: Cross-feature interactions for capturing relationships

### Model Architectures
- **Traditional ML**: Linear Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Deep Learning**: LSTM, GRU, Convolutional LSTM
- **Ensemble Methods**: Voting, weighted averaging, stacking

### Evaluation Framework
- **Comprehensive Metrics**: R², RMSE, MAE, MAPE, direction accuracy
- **Time Series Metrics**: Skill score, MASE, trend accuracy, seasonality correlation
- **Statistical Tests**: Residual normality, homoscedasticity, autocorrelation
- **Visualization**: Prediction plots, residual analysis, feature importance

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/paulo-cesar-pc/roller-mill-vibration.git
cd roller-mill-vibration
```

2. **Create a virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
# OR for development
pip install -e .[dev,jupyter,monitoring,advanced]
```

## Usage

### Quick Start

1. **Prepare your data**: Place CSV files in the `full_data/` directory (or configure path in `config/config.yaml`)

2. **Run the training pipeline**:
```bash
python scripts/train_models.py
```

3. **View results**: Check the `experiments/` and `models/` directories for outputs

### Configuration

Edit `config/config.yaml` to customize:
- Data paths and preprocessing options
- Feature engineering parameters
- Model architectures and hyperparameters
- Training and evaluation settings

### Advanced Usage

**Custom Model Training**:
```python
from src.models.trainer import ModelTrainer
from src.data.data_loader import DataLoader

# Load and process data
loader = DataLoader()
df, quality_report = loader.load_and_process()

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train, X_val, y_val)
```

**Feature Engineering Pipeline**:
```python
from src.features.feature_engineer import create_default_pipeline

# Create and apply feature engineering
pipeline = create_default_pipeline(config)
X_engineered = pipeline.fit_transform(X, y)
```

**Model Evaluation**:
```python
from src.evaluation.metrics import evaluate_model_comprehensive

# Comprehensive evaluation
results = evaluate_model_comprehensive(
    model, X_test, y_test, 
    save_plots=True, save_report=True
)
```

## Data Requirements

The system expects CSV files with:
- **Timestamp column**: DateTime values (configurable format)
- **Target variable**: `CM2_PV_VRM01_VIBRATION` (configurable)
- **Feature columns**: Sensor readings and process variables

Example data structure:
```csv
Timestamps,CM2_PV_VRM01_VIBRATION,CM2_PV_VRM01_POWER,CM2_PV_CLA01_SPEED,...
25/04/2024 00:00:00,5.2,1250,1450,...
25/04/2024 00:00:30,5.1,1248,1452,...
```

## Model Performance

Expected improvements over the original notebook approach:
- **R² Score**: 0.75+ (vs ~0.45 in original)
- **RMSE**: Significant reduction through ensemble methods
- **Generalization**: Better performance on unseen data through proper validation
- **Robustness**: Automated hyperparameter optimization and cross-validation

## Configuration Options

Key configuration sections:

### Data Processing
```yaml
data:
  target_column: "CM2_PV_VRM01_VIBRATION"
  train_split: 0.75
  validation_split: 0.15
  test_split: 0.10
```

### Feature Engineering
```yaml
features:
  lag_features:
    lags: [1, 2, 5, 10]
  rolling_features:
    windows: [5, 10, 30, 60]
    statistics: ["mean", "std", "min", "max"]
```

### Model Training
```yaml
training:
  optimization:
    method: "optuna"
    n_trials: 100
  cv_folds: 5
```

## License

This project is licensed under the MIT License.

## Next Steps

To run this improved system:

1. **Move your data**: Copy your CSV files to the `full_data/` directory
2. **Run training**: Execute `python scripts/train_models.py`
3. **Review results**: Check outputs in `experiments/` and `models/` directories
4. **Customize**: Modify `config/config.yaml` for your specific requirements

The new system should provide significantly better performance than the original notebook approach!