import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_data(data_dir='student', combine_datasets=True):
    """
    Load UCI Student Performance datasets.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the dataset files
    combine_datasets : bool
        If True, combine math and Portuguese datasets. If False, use only math dataset.
    
    Returns:
    --------
    df : pd.DataFrame
        Combined or single dataset
    """
    data_path = Path(data_dir)
    
    df_math = pd.read_csv(data_path / 'student-mat.csv', sep=';')
    df_por = pd.read_csv(data_path / 'student-por.csv', sep=';')
    
    if combine_datasets:
        df = pd.concat([df_math, df_por], ignore_index=True)
        df = df.drop_duplicates()
        print(f"Combined dataset: {len(df)} records")
    else:
        df = df_math.copy()
        print(f"Math dataset: {len(df)} records")
    
    return df


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with handled missing values
    """
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("Missing values found:")
        print(missing[missing > 0])
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
    else:
        print("No missing values found.")
    
    return df


def create_target_variables(df, dropout_threshold=10):
    """
    Create target variables for regression (G3) and classification (dropout).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    dropout_threshold : int
        Grade threshold below which student is considered at dropout risk
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with target variables
    y_regression : pd.Series
        Target for regression (G3 grades)
    y_classification : pd.Series
        Target for classification (dropout risk: 1 if G3 < threshold, 0 otherwise)
    """
    y_regression = df['G3'].copy()
    y_classification = (df['G3'] < dropout_threshold).astype(int)
    df['target_grade'] = y_regression
    df['target_dropout'] = y_classification
    
    print(f"\nTarget variable statistics:")
    print(f"Regression (G3): mean={y_regression.mean():.2f}, std={y_regression.std():.2f}")
    print(f"Classification (dropout): {y_classification.sum()} at risk ({y_classification.mean()*100:.1f}%)")
    
    return df, y_regression, y_classification


def prepare_features(df, target_cols=['G1', 'G2', 'G3', 'target_grade', 'target_dropout']):
    """
    Separate features from target variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_cols : list
        Columns to exclude from features
    
    Returns:
    --------
    X : pd.DataFrame
        Feature dataframe
    """
    X = df.drop(columns=target_cols, errors='ignore')
    return X


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Train and test splits
    """
    stratify = None
    if y.dtype in ['int64', 'int32', 'int'] or y.dtype.name == 'category':
        value_counts = y.value_counts()
        min_class_size = value_counts.min()
        if min_class_size >= 2:
            stratify = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    print(f"\nData split:")
    print(f"Train: {len(X_train)} samples")
    print(f"Test: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(data_dir='student', combine_datasets=True, dropout_threshold=10, test_size=0.2):
    """
    Complete preprocessing pipeline.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing dataset files
    combine_datasets : bool
        Whether to combine math and Portuguese datasets
    dropout_threshold : int
        Grade threshold for dropout classification
    test_size : float
        Proportion of test set
    
    Returns:
    --------
    dict : Dictionary containing all preprocessed data
    """
    df = load_data(data_dir, combine_datasets)
    df = handle_missing_values(df)
    df, y_regression, y_classification = create_target_variables(df, dropout_threshold)
    X = prepare_features(df)
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
        X, y_regression, test_size=test_size, random_state=42
    )
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(
        X, y_classification, test_size=test_size, random_state=42
    )
    
    return {
        'df': df,
        'X': X,
        'regression': {
            'X_train': X_train_reg,
            'X_test': X_test_reg,
            'y_train': y_train_reg,
            'y_test': y_test_reg
        },
        'classification': {
            'X_train': X_train_clf,
            'X_test': X_test_clf,
            'y_train': y_train_clf,
            'y_test': y_test_clf
        }
    }



