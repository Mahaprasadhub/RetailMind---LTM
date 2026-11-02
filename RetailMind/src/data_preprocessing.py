"""Dataset loading, inspection, and train/test split helpers."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIRECTORY / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"
RANDOM_STATE = 42


def inspect_dataset(dataframe):
    """Print a concise inspection report for the supplied dataset."""
    print("\nFirst five records:")
    print(dataframe.head().to_string(index=False))
    print(f"\nDataset shape: {dataframe.shape}")
    print("\nColumn data types:")
    print(dataframe.dtypes.to_string())
    print("\nMissing values by column:")
    print(dataframe.isna().sum().to_string())
    print(f"\nDuplicate records: {dataframe.duplicated().sum()}")
    print("\nChurn target distribution:")
    print(dataframe[TARGET_COLUMN].value_counts(dropna=False).to_string())
    print("\nDescriptive statistics:")
    print(dataframe.describe(include="all").to_string())

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in {IDENTIFIER_COLUMN, TARGET_COLUMN}
    ]
    numeric_columns = dataframe[feature_columns].select_dtypes(
        include="number"
    ).columns.tolist()
    categorical_columns = [
        column for column in feature_columns if column not in numeric_columns
    ]
    print(f"\nNumerical feature columns: {numeric_columns}")
    print(f"Categorical feature columns: {categorical_columns}")


def load_and_prepare_data(dataset_path=None):
    """Load the CSV, inspect it, and return model features and binary labels."""
    path = Path(dataset_path) if dataset_path is not None else DEFAULT_DATASET_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place "
            "'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the project's data/ folder."
        )

    dataframe = pd.read_csv(path)
    required_columns = {TARGET_COLUMN, IDENTIFIER_COLUMN, "TotalCharges"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    dataframe["TotalCharges"] = pd.to_numeric(
        dataframe["TotalCharges"], errors="coerce"
    )
    inspect_dataset(dataframe)

    target = dataframe[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        invalid_targets = dataframe.loc[target.isna(), TARGET_COLUMN].unique()
        raise ValueError(f"Unexpected values in Churn target: {invalid_targets}")

    features = dataframe.drop(columns=[IDENTIFIER_COLUMN, TARGET_COLUMN])
    return features, target.astype("int64")


def split_dataset(dataset_path=None):
    """Load the data and create a reproducible, stratified 80:20 split."""
    features, target = load_and_prepare_data(dataset_path)
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )
