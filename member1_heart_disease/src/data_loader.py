import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load heart disease dataset from CSV
    """
    df = pd.read_csv(file_path)
    return df


def basic_info(df: pd.DataFrame) -> None:
    """
    Print basic dataset information
    """
    print("Dataset Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())
