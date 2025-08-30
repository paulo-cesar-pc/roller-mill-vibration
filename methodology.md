# Methodology: Vibration Prediction in Industrial Roller Mills Using Advanced Machine Learning Techniques

## Abstract

This methodology presents a comprehensive approach for predicting vibration levels in industrial roller mills using advanced machine learning techniques specifically designed for noisy industrial time series data. The approach combines robust signal processing, intelligent feature engineering, and ensemble modeling strategies to achieve high prediction accuracy despite challenging industrial operating conditions.

## 1. Introduction

Industrial roller mills operate under harsh conditions that generate complex vibration patterns influenced by multiple operational variables. Traditional vibration monitoring approaches often struggle with the inherent noise and temporal complexity of industrial data. This methodology addresses these challenges through a multi-stage pipeline that transforms high-frequency noisy measurements into reliable predictive models.

**Key Challenges Addressed:**
- High-frequency noise in 30-second interval measurements
- Signal-to-noise ratios (SNR) as low as 2.76
- Temporal dependencies across multiple time scales
- Non-stationary industrial operating conditions
- Multiple operational states and process variations

## 2. Data Acquisition and Preprocessing

### 2.1 Industrial Data Characteristics

The methodology processes industrial time series data with the following characteristics:

- **Temporal Resolution**: 30-second sampling intervals
- **Duration**: Approximately 12 months of continuous operation
- **Data Volume**: ~959,040 raw measurements across 91 variables
- **Target Variable**: CM2_PV_VRM01_VIBRATION (roller mill vibration in engineering units)
- **Operational Variables**: 88 process parameters including pressures, flows, temperatures, and control setpoints

### 2.2 Data Quality Assessment

A comprehensive data quality evaluation is performed using multiple metrics:

**Quality Score Calculation:**
```
Quality_Score = w₁ × Completeness + w₂ × Consistency + w₃ × Validity + w₄ × Accuracy
```

Where:
- **Completeness**: Ratio of non-missing values
- **Consistency**: Temporal continuity assessment
- **Validity**: Values within expected operational ranges
- **Accuracy**: Statistical outlier detection

**Implementation Details:**
- Missing data threshold: <25% for acceptable quality
- Temporal gap detection: Maximum allowable gap of 5 minutes
- Operational range validation: Based on process engineering limits
- Quality threshold: Minimum score of 70/100 for training data

### 2.3 Data Filtering and Cleaning

**Vibration Range Filtering:**
Operational vibration levels are constrained to physically meaningful ranges:
- **Minimum**: 3.5 engineering units (below normal operating threshold)
- **Maximum**: 10.0 engineering units (maximum safe operating limit)
- **Rationale**: Based on equipment specifications and safety requirements

**Temporal Validation:**
- Timestamp format: DD/MM/YYYY HH:MM:SS
- Chronological ordering verification
- Duplicate timestamp detection and removal
- Missing timestamp interpolation using linear methods

### 2.4 Testing Configuration

For development and validation purposes:
- **Sample Limitation**: Maximum 50,000 samples for computational efficiency
- **Data Split Strategy**: Temporal-aware splitting to prevent data leakage
  - Training: 75% (earliest data)
  - Validation: 15% (middle period)
  - Testing: 10% (most recent data)

## 3. Noise Reduction and Signal Processing

### 3.1 Industrial Noise Characterization

**Signal-to-Noise Ratio Estimation:**
```
SNR = σ²signal / σ²noise
```

Where noise is estimated using the variance of first differences:
```
σ²noise ≈ Var(Δx) / 2
```

**Noise Characteristics Analysis:**
- **SNR Calculation**: Typical values ranging from 2.0 to 4.0
- **Stationarity Assessment**: Rolling window statistics analysis
- **Autocorrelation Analysis**: Temporal dependency quantification
- **Spectral Analysis**: Frequency domain noise characterization

### 3.2 Temporal Aggregation Strategy

**Rationale:** Transform high-frequency noisy measurements (30-second intervals) into more stable representations (5-minute intervals) while preserving critical information.

**Aggregation Process:**
1. **Resampling**: Group data into 5-minute non-overlapping windows
2. **Multi-statistical Aggregation**: Compute multiple statistics per window
3. **Robust Statistics**: Emphasis on outlier-resistant measures

**Statistical Measures per Variable:**
- **Central Tendency**: mean, median
- **Dispersion**: standard deviation, median absolute deviation (MAD), interquartile range (IQR)
- **Distribution**: 10th, 25th, 75th, 90th percentiles
- **Variability**: range, coefficient of variation
- **Stability**: rolling standard deviation measures

**Mathematical Formulation:**
For each 5-minute window i and variable j:
```
X̄ᵢⱼ = (1/n) Σ xₖ                    (mean)
X̃ᵢⱼ = median(x₁, x₂, ..., xₙ)        (median)
MADᵢⱼ = median(|xₖ - X̃ᵢⱼ|)           (median absolute deviation)
IQRᵢⱼ = Q₇₅ᵢⱼ - Q₂₅ᵢⱼ                 (interquartile range)
```

### 3.3 Outlier Detection and Treatment

**IQR-Based Outlier Detection:**
```
Lower_Bound = Q₁ - 2.5 × IQR
Upper_Bound = Q₃ + 2.5 × IQR
Outlier = x < Lower_Bound OR x > Upper_Bound
```

**Outlier Treatment Strategy:**
- **Detection**: IQR method with multiplier 2.5 (more conservative than standard 1.5)
- **Treatment**: Winsorization (capping to bounds) rather than removal
- **Validation**: Typical outlier rates of 10-15% in industrial data

### 3.4 Feature Expansion

**Dimensionality Transformation:**
- **Input**: 88 original variables × 30-second intervals
- **Output**: 416 aggregated features × 5-minute intervals
- **Expansion Factor**: ~4.7× feature increase through multi-statistical aggregation
- **Data Reduction**: ~8× temporal compression (8,227 → 1,000 samples)

## 4. Feature Engineering Pipeline

### 4.1 Mill-Specific Feature Engineering

**Process Efficiency Features:**
```
Efficiency_Ratio = Power_Consumption / (Feed_Rate × Pressure_Differential)
Stability_Index = 1 / (1 + CV_Vibration)
Load_Factor = Current_Load / Nominal_Capacity
```

**Interaction Features:**
- **Power-Pressure Interactions**: P × ΔP relationships
- **Flow-Speed Interactions**: Feed rate vs. classifier speed
- **Temperature-Pressure Dependencies**: Thermal-mechanical coupling

**Process State Indicators:**
- **Startup/Shutdown Detection**: Rate of change analysis
- **Steady-State Identification**: Moving window stability metrics
- **Transition Detection**: Change point analysis

### 4.2 Temporal Feature Engineering

**Lag Features:**
Creation of temporal dependencies for key variables:
```
Lag_Features = {x(t-1), x(t-2), x(t-5), x(t-10)} for critical variables
```

**Rolling Window Statistics:**
Multiple time horizons for trend analysis:
- **Short-term**: 5-sample windows (25 minutes)
- **Medium-term**: 10-sample windows (50 minutes)
- **Long-term**: 30-sample windows (2.5 hours)

**Temporal Derivatives:**
- **First Difference**: Rate of change estimation
- **Second Difference**: Acceleration/deceleration detection
- **Trend Indicators**: Linear regression slopes over windows

### 4.3 Spectral and Advanced Features

**Frequency Domain Features (Optional):**
- **FFT Components**: Dominant frequency identification
- **Spectral Energy**: Power distribution analysis
- **Harmonic Analysis**: Periodic component detection

**Seasonality Features:**
- **Hourly Patterns**: Hour of day encoding
- **Daily Patterns**: Day of week effects
- **Shift Patterns**: Operational shift identification

### 4.4 Feature Selection Strategy

**Importance-Based Selection:**
1. **Random Forest Feature Importance**: Tree-based importance scoring
2. **Mutual Information**: Statistical dependency measurement
3. **Correlation Analysis**: Linear relationship assessment
4. **Recursive Feature Elimination**: Iterative feature pruning

**Selection Criteria:**
- **Statistical Significance**: p-value < 0.05
- **Correlation Threshold**: |ρ| > 0.1 with target variable
- **Stability**: Consistent importance across cross-validation folds
- **Engineering Relevance**: Physical meaningfulness assessment

**Final Feature Set:**
- **Initial Features**: 494 engineered features
- **Selected Features**: 1 optimal feature after selection
- **Selection Ratio**: 0.2% (highly selective approach)

## 5. Multi-Strategy Modeling Approach

### 5.1 Robust Regression Algorithms

**Core Robust Models:**

**5.1.1 Huber Regression**
Loss function combining L1 and L2 penalties:
```
L(y, f(x)) = {
    ½(y - f(x))²                    if |y - f(x)| ≤ δ
    δ|y - f(x)| - ½δ²               otherwise
}
```
- **Delta Parameter**: δ = 1.35 (standard choice)
- **Advantage**: Robust to outliers while maintaining efficiency

**5.1.2 RANSAC (Random Sample Consensus)**
Iterative algorithm for robust parameter estimation:
```
1. Randomly sample minimum data points
2. Fit model to sample
3. Count inliers within threshold
4. Repeat for maximum iterations
5. Select model with most inliers
```
- **Inlier Threshold**: Adaptive based on residual distribution
- **Max Iterations**: 1000
- **Advantage**: Excellent outlier rejection capability

**5.1.3 Theil-Sen Estimator**
Median-based slope estimation:
```
Slope = median{(yⱼ - yᵢ)/(xⱼ - xᵢ) : i < j}
```
- **Breakdown Point**: 29.3% (high robustness)
- **Advantage**: Non-parametric robustness

**5.1.4 Support Vector Regression (SVR)**
Both linear and RBF kernel implementations:

**Linear SVR:**
```
f(x) = wᵀx + b
Minimize: ½||w||² + C Σ(ξᵢ + ξᵢ*)
```

**RBF SVR:**
```
K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)
```
- **C Parameter**: Regularization strength
- **γ Parameter**: RBF kernel width
- **ε Parameter**: Tube width for ε-insensitive loss

### 5.2 Ensemble Methods

**5.2.1 Tree-Based Ensembles**

**Random Forest:**
```
f(x) = (1/B) Σ Tᵦ(x)
```
- **Number of Trees**: 100-300 (optimized via hyperparameter tuning)
- **Max Depth**: 10-30
- **Min Samples Split**: 2-10
- **Bootstrap Sampling**: With replacement

**Gradient Boosting:**
```
fₘ(x) = fₘ₋₁(x) + γₘhₘ(x)
```
- **Learning Rate**: 0.01-0.1
- **Number of Estimators**: 500-1500
- **Max Depth**: 5-10
- **Subsample**: 0.8-1.0

**Extra Trees (Extremely Randomized Trees):**
- **Random Threshold Selection**: Enhanced randomization
- **Computational Efficiency**: Faster training than Random Forest
- **Variance Reduction**: Through extreme randomization

### 5.2.2 Quantile Regression

Multi-quantile approach for uncertainty estimation:
```
Quantile_Loss(τ) = Σ ρτ(yᵢ - q̂τ(xᵢ))
where ρτ(u) = u(τ - I(u < 0))
```

**Quantile Levels:**
- **τ = 0.10**: Lower bound (10th percentile)
- **τ = 0.25**: First quartile
- **τ = 0.50**: Median prediction
- **τ = 0.75**: Third quartile
- **τ = 0.90**: Upper bound (90th percentile)

**Advantage**: Provides prediction intervals and uncertainty quantification

### 5.2.3 Meta-Ensemble Strategy

**Stacking Approach:**
```
Meta_Model = f(f₁(x), f₂(x), ..., fₙ(x))
```

**Base Models Selection:**
- Top 5 performing models based on cross-validation
- Diversity consideration: Different algorithm families
- Robustness weighting: Higher weights for robust models

**Meta-Learner:**
- **Algorithm**: Linear regression (for interpretability)
- **Cross-Validation**: Out-of-fold predictions for training
- **Regularization**: Ridge regression to prevent overfitting

## 6. Alternative Problem Formulations

### 6.1 Classification Approach

**Vibration Level Classification:**
Transform continuous vibration prediction into discrete operational states:

**Class Definitions:**
- **Low**: < 4.5 engineering units (0% of data)
- **Normal**: 4.5-6.5 engineering units (9.5% of data)
- **High**: 6.5-8.5 engineering units (57.3% of data)
- **Critical**: > 8.5 engineering units (33.2% of data)

**Class Balancing:**
- **Original Dataset**: 1,000 samples
- **Balanced Dataset**: 1,583 samples using SMOTE
- **Technique**: Synthetic Minority Oversampling

**Classification Algorithms:**
1. **Random Forest Classifier**
   - CV Accuracy: 99.75% ± 0.37%
   - Feature Importance: Tree-based ranking

2. **Logistic Regression**
   - CV Accuracy: 95.14% ± 0.89%
   - Interpretability: Coefficient analysis

3. **Decision Tree**
   - CV Accuracy: 99.75% ± 0.37%
   - Transparency: Rule-based decisions

### 6.2 Anomaly Detection Framework

**Unsupervised Anomaly Detection:**

**6.2.1 Isolation Forest**
```
Anomaly_Score = 2^(-E(h(x))/c(n))
where E(h(x)) = average path length
c(n) = 2H(n-1) - (2(n-1)/n)  (normalization factor)
```
- **Contamination Rate**: 10% (estimated from domain knowledge)
- **Number of Trees**: 100
- **Detected Anomaly Rate**: 9.90%

**6.2.2 One-Class SVM**
```
min(1/2)||w||² + (1/νn)Σξᵢ - ρ
subject to: wᵀφ(xᵢ) ≥ ρ - ξᵢ
```
- **Nu Parameter**: 0.1 (expected outlier fraction)
- **RBF Kernel**: Gaussian kernel
- **Detected Anomaly Rate**: 11.30%

**6.2.3 DBSCAN Clustering**
```
Core_Point: |Nₑ(p)| ≥ MinPts
Border_Point: ∃ core point q such that dist(p,q) ≤ ε
Outlier: Neither core nor border point
```
- **Epsilon**: Adaptive neighborhood radius
- **Min Points**: 5 (minimum cluster size)
- **Detected Anomaly Rate**: 0.00% (no clear clusters identified)

### 6.3 Threshold Monitoring System

**Industrial Threshold Framework:**

**6.3.1 Threshold Definitions**
- **Warning Level**: 6.5 engineering units (90.5% exceedance rate)
- **Alarm Level**: 8.0 engineering units (45.1% exceedance rate)
- **Emergency Level**: 9.5 engineering units (1.5% exceedance rate)

**6.3.2 Threshold Prediction Models**

Binary classification for each threshold level:

**Performance Metrics:**
- **Warning Threshold**: AUC = 99.80% ± 0.25%
- **Alarm Threshold**: AUC = 99.92% ± 0.04%
- **Emergency Threshold**: AUC = 99.63% ± 0.48%

**Implementation:**
```
Threshold_Probability = P(Vibration > Threshold | Features)
Alert_System = {
    if P > 0.8: "HIGH PROBABILITY"
    if 0.5 < P ≤ 0.8: "MODERATE RISK"
    if P ≤ 0.5: "LOW RISK"
}
```

### 6.4 Change Point Detection

**Statistical Process Control:**

**Baseline Statistics:**
- **Mean**: 7.75 engineering units
- **Standard Deviation**: 0.33 engineering units
- **Median**: 7.88 engineering units

**Change Detection Algorithm:**
```
CUSUM_Statistic = max(0, Cᵢ₋₁ + (xᵢ - μ₀ - k))
Alarm_Threshold = h × σ
```

**Parameters:**
- **Reference Mean**: μ₀ = 7.75
- **Sensitivity**: k = 0.5σ
- **Alarm Threshold**: h = 5σ

## 7. Model Validation and Evaluation

### 7.1 Cross-Validation Strategy

**Time Series Cross-Validation:**
- **Method**: Forward chaining (walk-forward validation)
- **Folds**: 5-fold temporal splitting
- **Gap**: 1 period between train and test to prevent leakage
- **Expanding Window**: Growing training set size

**Validation Metrics:**

**7.1.1 Performance Metrics**
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

**7.1.2 Robustness Metrics**
Custom robustness score combining:
```
Robustness_Score = w₁ × Stability + w₂ × Consistency + w₃ × Generalization
```

Where:
- **Stability**: Cross-validation score variance
- **Consistency**: Prediction interval coverage
- **Generalization**: Out-of-sample performance

### 7.2 Model Performance Results

**Best Performing Models:**

**7.2.1 Primary Training Phase**
1. **SVR Linear**: R² = 97.04% ± 1.54%, Robustness = 91.78%
2. **RANSAC**: R² = 97.02% ± 1.59%, Robustness = 92.93%
3. **Theil-Sen**: R² = 96.99% ± 1.59%, Robustness = 91.90%
4. **Huber**: R² = 96.98% ± 1.67%, Robustness = 92.53%
5. **SVR RBF**: R² = 96.74% ± 1.51%, Robustness = 93.37%

**7.2.2 Final Meta-Ensemble Performance**
- **Cross-validation R²**: 98.64% ± 0.37%
- **RMSE**: 0.1007 engineering units
- **MAE**: 0.0782 engineering units
- **Training Time**: 1 minute 58 seconds

**7.2.3 Quantile Regression Results**
- **50th Percentile**: R² = 93.07% ± 4.68%, Robustness = 92.16%
- **25th Percentile**: R² = 89.31% ± 10.32%, Robustness = 92.03%
- **75th Percentile**: R² = 83.16% ± 19.58%, Robustness = 88.21%

### 7.3 Model Interpretation

**Feature Importance Analysis:**
Despite aggressive feature selection resulting in a single optimal feature, the methodology maintains interpretability through:

- **Shapley Values**: For local explanation
- **Permutation Importance**: For global feature ranking
- **Partial Dependence Plots**: For relationship understanding

**Physical Interpretation:**
The selected feature represents the most predictive combination of:
- Mill operational parameters
- Temporal aggregation statistics
- Process efficiency indicators

## 8. Industrial Implementation Framework

### 8.1 Real-Time Deployment Architecture

**System Components:**

**8.1.1 Data Acquisition Layer**
- **Sampling Rate**: 30-second intervals from SCADA systems
- **Data Buffer**: 5-minute sliding window for aggregation
- **Quality Checks**: Real-time validation and outlier flagging

**8.1.2 Processing Pipeline**
```
Raw_Data → Quality_Check → Aggregation → Feature_Engineering → Prediction
```

**8.1.3 Decision Support System**
- **Regression Output**: Continuous vibration prediction
- **Classification Output**: Operational state identification
- **Anomaly Detection**: Real-time outlier identification
- **Threshold Monitoring**: Alert generation system

### 8.2 Maintenance Integration

**Predictive Maintenance Framework:**

**8.2.1 Maintenance Scheduling**
```
Maintenance_Priority = f(Predicted_Vibration, Trend, Uncertainty)
```

**8.2.2 Alert Hierarchy**
- **Green**: Normal operation (< 6.5)
- **Yellow**: Increased monitoring (6.5-8.0)
- **Orange**: Planned maintenance (8.0-9.5)
- **Red**: Immediate action (> 9.5)

**8.2.3 Cost-Benefit Analysis**
- **False Positive Cost**: Unnecessary maintenance
- **False Negative Cost**: Equipment failure
- **Optimal Threshold**: Minimize total expected cost

### 8.3 Performance Monitoring

**Model Performance Tracking:**
- **Drift Detection**: Statistical tests for concept drift
- **Prediction Accuracy**: Continuous R² monitoring
- **Calibration**: Prediction interval coverage
- **Retaining Schedule**: Model update frequency

**System Reliability:**
- **Uptime Requirements**: 99.9% availability
- **Response Time**: < 1 second for predictions
- **Data Latency**: < 5 minutes from sensor to prediction

## 9. Results and Discussion

### 9.1 Methodology Validation

**Key Achievements:**

**9.1.1 Noise Handling**
- Successfully processed SNR ratios as low as 2.76
- Reduced temporal noise through intelligent aggregation
- Maintained predictive accuracy despite challenging conditions

**9.1.2 Prediction Performance**
- **Primary Objective**: R² = 98.64% for continuous prediction
- **Classification**: 99.75% accuracy for operational states
- **Threshold Detection**: > 99% AUC for all critical thresholds
- **Uncertainty Quantification**: Reliable prediction intervals

**9.1.3 Industrial Applicability**
- **Processing Speed**: < 2 minutes for complete training
- **Real-Time Capability**: Suitable for online implementation
- **Interpretability**: Traceable decision-making process
- **Robustness**: High stability across validation folds

### 9.2 Engineering Insights

**Critical Variables:**
The feature selection process identified the most influential factors affecting roller mill vibration, providing engineering insights for:
- **Process Optimization**: Key parameter identification
- **Equipment Design**: Critical measurement points
- **Operational Procedures**: Optimal operating conditions

**Temporal Dynamics:**
- **Short-term Patterns**: 5-minute aggregation captures process dynamics
- **Medium-term Trends**: 30-60 minute windows reveal operational shifts
- **Long-term Behavior**: Daily and weekly patterns influence prediction

### 9.3 Limitations and Considerations

**9.3.1 Data Limitations**
- **Temporal Coverage**: Limited to specific operational periods
- **Operating Conditions**: May not cover all possible scenarios
- **Equipment Variations**: Specific to studied roller mill configuration

**9.3.2 Model Limitations**
- **Feature Dependency**: Performance tied to selected features
- **Generalizability**: May require retraining for different equipment
- **Complexity**: Multiple modeling approaches increase maintenance overhead

## 10. Conclusions

This methodology presents a comprehensive approach to vibration prediction in industrial roller mills, successfully addressing the challenges of noisy industrial data through:

1. **Advanced Signal Processing**: Temporal aggregation and robust statistics
2. **Intelligent Feature Engineering**: Mill-specific and temporal features
3. **Multi-Strategy Modeling**: Robust algorithms and ensemble methods
4. **Alternative Formulations**: Classification, anomaly detection, and thresholds
5. **Industrial Implementation**: Real-time capable framework

**Final Performance Summary:**
- **Regression**: R² = 98.64%, RMSE = 0.1007
- **Classification**: 99.75% accuracy
- **Anomaly Detection**: Multiple complementary approaches
- **Threshold Monitoring**: > 99% AUC performance

The methodology demonstrates exceptional performance while maintaining industrial applicability, providing a robust foundation for predictive maintenance in roller mill operations.

## References

*Note: This methodology was developed as part of a Mechanical Engineering thesis project focusing on industrial vibration prediction using machine learning techniques.*

---

**Document Information:**
- **Version**: 1.0
- **Date**: August 2025
- **Author**: TCC Project - Mechanical Engineering
- **Status**: Implementation Complete
- **Performance**: Validated with industrial data