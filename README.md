# Student Performance / Dropout Prediction with Machine Learning

## Project Overview

This project implements machine learning models to predict student final grades and identify at-risk students using the UCI Student Performance dataset. The solution includes a regression model for grade prediction and a classification model for dropout risk identification.

**Key Results:**
- **Regression Model:** RMSE 3.65, R² 0.14 (predicts grades within ±3.6 points on average)
- **Classification Model:** ROC-AUC 0.69, Precision 50%, Recall 43%, F1 0.47, Accuracy 78%


## Project Structure

```
ai_new_proj/
├── main.py                          # Main execution script - orchestrates entire pipeline
├── src/                             # Source code modules
│   ├── __init__.py
│   ├── data_processing.py          # Data loading, preprocessing, target variable creation
│   ├── feature_engineering.py      # Feature creation, encoding, transformation
│   ├── models.py                   # XGBoost model training (regression & classification)
│   └── evaluation.py               # Model evaluation, metrics, visualizations
├── models/                          # Trained model artifacts
│   ├── regression_model.pkl        # Saved regression model
│   └── classification_model.pkl    # Saved classification model
├── results/                         # Evaluation results and visualizations
│   ├── classification/             # Classification model results
│   │   ├── classification_confusion_matrix.png
│   │   ├── classification_roc_curve.png
│   │   ├── classification_probability_distribution.png
│   │   └── classification_report.csv
│   ├── regression/                 # Regression model results
│   │   ├── regression_prediction_vs_actual.png
│   │   ├── regression_residuals.png
│   │   └── regression_residuals_distribution.png
│   ├── evaluation_summary.csv      # Summary of all metrics
│   └── model_comparison_metrics.csv
└── student/                         # Dataset files
    ├── student-mat.csv             # Mathematics course data (395 students)
    ├── student-por.csv             # Portuguese course data (649 students)
    └── student.txt                  # Dataset documentation
```

## Code Navigation Guide

### Entry Point: `main.py`

The main script orchestrates the complete ML pipeline:

1. **Data Loading** (Line 18-24): Calls `preprocess_pipeline()` to load and prepare data
2. **Feature Engineering** (Line 26-47): Applies feature engineering to both regression and classification datasets
3. **Model Training** (Line 49-68): Trains XGBoost regression and classification models
4. **Model Saving** (Line 70-75): Saves trained models to disk
5. **Evaluation** (Line 77-88): Generates comprehensive evaluation reports

**To run:** `python main.py`

### Module 1: `src/data_processing.py`

**Purpose:** Data loading, cleaning, and target variable creation

**Key Functions:**
- `load_data()`: Loads CSV files (semicolon-delimited), combines datasets
- `handle_missing_values()`: Handles missing data (none found in this dataset)
- `create_target_variables()`: Creates two targets:
  - Regression: G3 (final grade, continuous)
  - Classification: Dropout risk (binary: G3 < 10)
- `prepare_features()`: Separates features from target variables
- `split_data()`: Train/test split with stratification for classification
- `preprocess_pipeline()`: Complete preprocessing workflow

**Key Details:**
- Combines math and Portuguese datasets (1,044 total students)
- Creates binary dropout classification (threshold = 10)
- 80/20 train/test split

### Module 2: `src/feature_engineering.py`

**Purpose:** Feature creation, encoding, and transformation

**Key Functions:**
- `encode_categorical_features()`: One-hot encoding for categorical variables
- `create_derived_features()`: Creates 48 features from 33 original:
  - Grade features: averages, improvements, trends
  - Parent education: averages, max, sum
  - Support scores: binary conversions, total support
  - Social features: activity scores, alcohol consumption
  - Interaction features: failure × absences, family quality
- `select_features()`: Optional feature selection (correlation-based)
- `scale_features()`: Optional scaling (not used for tree-based models)
- `feature_engineering_pipeline()`: Complete feature engineering workflow

**Key Details:**
- Expands 33 original features to 48 engineered features
- One-hot encoding creates additional binary features
- No scaling applied (XGBoost handles this internally)

### Module 3: `src/models.py`

**Purpose:** Model training with XGBoost

**Key Functions:**
- `train_regression_model()`: Trains XGBoost regressor for grade prediction
  - Default parameters: n_estimators=100, max_depth=6, learning_rate=0.1
  - Optional hyperparameter tuning with RandomizedSearchCV
  - Returns model and metrics (RMSE, MAE, R²)
- `train_classification_model()`: Trains XGBoost classifier for dropout prediction
  - Handles class imbalance with `scale_pos_weight`
  - Default parameters similar to regression
  - Returns model and metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- `save_model()` / `load_model()`: Model persistence utilities

**Key Details:**
- XGBoost chosen for strong performance on structured data
- Both models use same algorithm with task-specific objectives
- Cross-validation included for robust evaluation

### Module 4: `src/evaluation.py`

**Purpose:** Model evaluation and visualization

**Key Functions:**
- `evaluate_regression_model()`: 
  - Calculates RMSE, MAE, R², MAPE
  - Creates prediction vs actual scatter plot
  - Generates residuals plots
- `evaluate_classification_model()`:
  - Calculates accuracy, precision, recall, F1, ROC-AUC
  - Creates confusion matrix heatmap
  - Generates ROC curve
  - Creates probability distribution plots
- `generate_evaluation_report()`: Complete evaluation workflow for both models

**Key Details:**
- All visualizations saved as PNG files (300 DPI)
- Metrics saved as CSV files
- Comprehensive evaluation suitable for academic presentation

## Dataset Information

**Source:** UCI Machine Learning Repository - Student Performance Dataset

**Contents:**
- 395 students (Mathematics course)
- 649 students (Portuguese course)
- 382 students appear in both datasets
- 33 features: demographics, academic history, family support, social factors
- Target: G1, G2 (period grades), G3 (final grade, 0-20 scale)

**Key Features:**
- Demographics: school, sex, age, address, family size
- Academic: study time, past failures, absences, support received
- Family: parent education, jobs, family relationships
- Social: activities, internet access, alcohol consumption, romantic relationships


## Results Location

All results are saved in the `results/` directory:

**Classification Results:**
- `classification/classification_confusion_matrix.png` - Performance breakdown
- `classification/classification_roc_curve.png` - Model discrimination
- `classification/classification_probability_distribution.png` - Prediction confidence
- `classification/classification_report.csv` - Detailed metrics

**Regression Results:**
- `regression/regression_prediction_vs_actual.png` - Prediction accuracy
- `regression/regression_residuals.png` - Error analysis
- `regression/regression_residuals_distribution.png` - Error distribution

**Summary Files:**
- `evaluation_summary.csv` - All metrics in one table
- `model_comparison_metrics.csv` - Side-by-side comparison


