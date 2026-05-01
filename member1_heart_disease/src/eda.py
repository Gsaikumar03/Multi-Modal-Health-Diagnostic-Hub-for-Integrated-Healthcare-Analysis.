import pandas as pd


def target_distribution(df: pd.DataFrame, target_col: str) -> None:
    """
    Print target variable distribution
    """
    print("\nTarget Variable Distribution:")
    print(df[target_col].value_counts())
    print("\nTarget Variable Percentage:")
    print(df[target_col].value_counts(normalize=True) * 100)


def numerical_summary(df: pd.DataFrame) -> None:
    """
    Display statistical summary for numerical features
    """
    print("\nNumerical Features Summary:")
    print(df.describe())


def categorical_summary(df: pd.DataFrame) -> None:
    """
    Display unique values for categorical features
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(categorical_cols) == 0:
        print("\nNo categorical columns detected.")
        return

    print("\nCategorical Feature Summary:")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts())
