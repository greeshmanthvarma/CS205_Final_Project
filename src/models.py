import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer
import numpy as np
import joblib
from pathlib import Path


def train_regression_model(X_train, y_train, X_test=None, y_test=None, 
                          use_grid_search=False, cv=5, random_state=42):
    """
    Train XGBoost regression model for grade prediction.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target (G3 grades)
    X_test : pd.DataFrame or np.array, optional
        Test features
    y_test : pd.Series or np.array, optional
        Test target
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
    
    Returns:
    --------
    model : xgb.XGBRegressor
        Trained model
    results : dict
        Dictionary with model performance metrics
    """
    print("\nTraining Regression Model (Grade Prediction)...")
    
    base_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'n_jobs': -1
    }
    
    if use_grid_search:
        # Hyperparameter grid for tuning
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        base_model = xgb.XGBRegressor(**base_params)
        
        grid_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
    else:
        model = xgb.XGBRegressor(**base_params)
        model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, 
                               scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    
    print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
    
    results = {
        'model': model,
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std()
    }
    
    if X_test is not None and y_test is not None:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        y_pred = model.predict(X_test)
        results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        results['test_mae'] = mean_absolute_error(y_test, y_pred)
        results['test_r2'] = r2_score(y_test, y_pred)
        print(f"Test RMSE: {results['test_rmse']:.4f}")
        print(f"Test MAE: {results['test_mae']:.4f}")
        print(f"Test R²: {results['test_r2']:.4f}")
    
    return model, results


def train_classification_model(X_train, y_train, X_test=None, y_test=None,
                               use_grid_search=False, cv=5, random_state=42):
    """
    Train XGBoost classification model for dropout prediction.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.array
        Training features
    y_train : pd.Series or np.array
        Training target (dropout: 1 or 0)
    X_test : pd.DataFrame or np.array, optional
        Test features
    y_test : pd.Series or np.array, optional
        Test target
    use_grid_search : bool
        Whether to use grid search for hyperparameter tuning
    cv : int
        Number of cross-validation folds
    random_state : int
        Random seed
    
    Returns:
    --------
    model : xgb.XGBClassifier
        Trained model
    results : dict
        Dictionary with model performance metrics
    """
    print("\nTraining Classification Model (Dropout Prediction)...")
    
    class_counts = np.bincount(y_train)
    if len(class_counts) == 2:
        scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        print(f"Class distribution: {class_counts[0]} (no dropout), {class_counts[1]} (dropout)")
        print(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0
    
    base_params = {
        'objective': 'binary:logistic',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'random_state': random_state,
        'n_jobs': -1,
        'eval_metric': 'logloss'
    }
    
    if use_grid_search:
        # Hyperparameter grid for tuning
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        base_model = xgb.XGBClassifier(**base_params)
        
        grid_search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=random_state,
            verbose=1
        )
        
        print("Performing hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
    else:
        model = xgb.XGBClassifier(**base_params)
        model.fit(X_train, y_train)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Cross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    results = {
        'model': model,
        'cv_roc_auc_mean': cv_scores.mean(),
        'cv_roc_auc_std': cv_scores.std()
    }
    
    if X_test is not None and y_test is not None:
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, roc_auc_score, confusion_matrix)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results['test_accuracy'] = accuracy_score(y_test, y_pred)
        results['test_precision'] = precision_score(y_test, y_pred, zero_division=0)
        results['test_recall'] = recall_score(y_test, y_pred, zero_division=0)
        results['test_f1'] = f1_score(y_test, y_pred, zero_division=0)
        results['test_roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Test Precision: {results['test_precision']:.4f}")
        print(f"Test Recall: {results['test_recall']:.4f}")
        print(f"Test F1: {results['test_f1']:.4f}")
        print(f"Test ROC-AUC: {results['test_roc_auc']:.4f}")
    
    return model, results


def save_model(model, filepath, model_type='regression'):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor or xgb.XGBClassifier
        Trained model
    filepath : str or Path
        Path to save model
    model_type : str
        'regression' or 'classification'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to saved model
    
    Returns:
    --------
    model : xgb.XGBRegressor or xgb.XGBClassifier
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

