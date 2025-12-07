import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def encode_categorical_features(X_train, X_test=None, encoding_method='onehot'):
    """
    Encode categorical features.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame, optional
        Test features
    encoding_method : str
        'onehot' for one-hot encoding, 'label' for label encoding
    
    Returns:
    --------
    X_train_encoded : pd.DataFrame
        Encoded training features
    X_test_encoded : pd.DataFrame, optional
        Encoded test features
    encoders : dict
        Dictionary of encoders for each categorical column
    """
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy() if X_test is not None else None
    
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    encoders = {}
    
    if encoding_method == 'onehot':
        X_train_encoded = pd.get_dummies(X_train_encoded, columns=categorical_cols, prefix=categorical_cols)
        if X_test_encoded is not None:
            X_test_encoded = pd.get_dummies(X_test_encoded, columns=categorical_cols, prefix=categorical_cols)
            train_cols = set(X_train_encoded.columns)
            test_cols = set(X_test_encoded.columns)
            for col in train_cols - test_cols:
                X_test_encoded[col] = 0
            X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    elif encoding_method == 'label':
        for col in categorical_cols:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
            encoders[col] = le
            if X_test_encoded is not None:
                test_values = X_test_encoded[col].astype(str)
                unseen = set(test_values.unique()) - set(le.classes_)
                if unseen:
                    most_common = le.classes_[0]
                    test_values = test_values.replace(list(unseen), most_common)
                X_test_encoded[col] = le.transform(test_values)
    
    return X_train_encoded, X_test_encoded, encoders


def create_derived_features(df):
    """
    Create derived features from existing ones.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with original features
    
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with additional derived features
    """
    df = df.copy()
    
    if 'G1' in df.columns and 'G2' in df.columns:
        df['avg_grade_G1_G2'] = (df['G1'] + df['G2']) / 2
        df['grade_improvement'] = df['G2'] - df['G1']
        df['grade_trend'] = np.where(df['grade_improvement'] > 0, 'improving', 
                                    np.where(df['grade_improvement'] < 0, 'declining', 'stable'))
    
    if 'Medu' in df.columns and 'Fedu' in df.columns:
        df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
        df['parent_edu_max'] = df[['Medu', 'Fedu']].max(axis=1)
        df['parent_edu_sum'] = df['Medu'] + df['Fedu']
    
    support_cols = []
    if 'schoolsup' in df.columns:
        support_cols.append('schoolsup')
    if 'famsup' in df.columns:
        support_cols.append('famsup')
    
    if support_cols:
        for col in support_cols:
            if df[col].dtype == 'object':
                df[f'{col}_binary'] = (df[col] == 'yes').astype(int)
        
        binary_cols = [f'{col}_binary' for col in support_cols]
        if binary_cols:
            df['total_support'] = df[binary_cols].sum(axis=1)
    
    if 'studytime' in df.columns:
        df['studytime_squared'] = df['studytime'] ** 2
    
    if 'absences' in df.columns:
        df['absences_log'] = np.log1p(df['absences'])
        df['high_absences'] = (df['absences'] > df['absences'].median()).astype(int)
    
    if 'Dalc' in df.columns and 'Walc' in df.columns:
        df['alcohol_avg'] = (df['Dalc'] + df['Walc']) / 2
        df['alcohol_total'] = df['Dalc'] + df['Walc']
    
    social_cols = []
    if 'goout' in df.columns:
        social_cols.append('goout')
    if 'freetime' in df.columns:
        social_cols.append('freetime')
    
    if social_cols:
        df['social_activity'] = df[social_cols].mean(axis=1) if len(social_cols) > 1 else df[social_cols[0]]
    
    if 'age' in df.columns:
        df['age_squared'] = df['age'] ** 2
        df['is_older'] = (df['age'] > df['age'].median()).astype(int)
    
    if 'failures' in df.columns:
        df['has_failures'] = (df['failures'] > 0).astype(int)
        if 'absences' in df.columns:
            df['failure_risk'] = df['failures'] * df['absences']
        else:
            df['failure_risk'] = df['failures']
    
    if 'famsize' in df.columns and 'famrel' in df.columns:
        df['famsize_binary'] = (df['famsize'] == 'GT3').astype(int)
        df['family_quality'] = df['famsize_binary'] * df['famrel']
    
    print(f"Created {len([col for col in df.columns if col not in ['G1', 'G2', 'G3']])} features")
    
    return df


def select_features(X, y, method='correlation', top_k=20):
    """
    Select most important features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    method : str
        'correlation' for correlation-based, 'importance' for model-based
    top_k : int
        Number of top features to select
    
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    if method == 'correlation':
        # Calculate correlation with target
        if X.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(top_k).index.tolist()
        else:
            # If mixed types, use all numeric features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            correlations = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(top_k).index.tolist()
    else:
        # Use all features if method not specified
        selected_features = X.columns.tolist()
    
    print(f"Selected {len(selected_features)} features using {method} method")
    
    return selected_features


def scale_features(X_train, X_test=None):
    """
    Scale numerical features.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame, optional
        Test features
    
    Returns:
    --------
    X_train_scaled : pd.DataFrame
        Scaled training features
    X_test_scaled : pd.DataFrame, optional
        Scaled test features
    scaler : StandardScaler
        Fitted scaler
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, None, scaler


def feature_engineering_pipeline(X_train, X_test, y_train, create_derived=True, 
                                  encode_categorical=True, scale_features_flag=False, 
                                  feature_selection=False, top_k=30):
    """
    Complete feature engineering pipeline.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : pd.Series
        Training target (for feature selection)
    create_derived : bool
        Whether to create derived features
    encode_categorical : bool
        Whether to encode categorical features
    scale_features_flag : bool
        Whether to scale numerical features
    feature_selection : bool
        Whether to perform feature selection
    top_k : int
        Number of features to select
    
    Returns:
    --------
    dict : Dictionary containing processed features and transformers
    """
    print("Starting feature engineering pipeline...")
    
    # Create derived features
    if create_derived:
        X_train = create_derived_features(X_train)
        X_test = create_derived_features(X_test)
        print("✓ Derived features created")
    
    # Encode categorical features
    if encode_categorical:
        X_train, X_test, encoders = encode_categorical_features(X_train, X_test, encoding_method='onehot')
        print("✓ Categorical features encoded")
    else:
        encoders = {}
    
    # Feature selection
    selected_features = None
    if feature_selection:
        selected_features = select_features(X_train, y_train, method='correlation', top_k=top_k)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        print("✓ Feature selection completed")
    
    # Scale features (optional, usually not needed for tree-based models)
    scaler = None
    if scale_features_flag:
        X_train, X_test, scaler = scale_features(X_train, X_test)
        print("✓ Features scaled")
    
    print(f"Final feature shape: {X_train.shape}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'encoders': encoders,
        'scaler': scaler,
        'selected_features': selected_features
    }

