import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Preprocess heart dataset
    - Handle missing values
    - Encode categorical variables
    - Scale numerical features
    """

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Encode target column
    df[target_column] = df[target_column].map({
        "Presence": 1,
        "Absence": 0
    })

    # Separate target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encoding for categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
