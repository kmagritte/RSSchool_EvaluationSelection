from pathlib import Path
from typing import Tuple
import click
import pandas as pd
from sklearn.model_selection import train_test_split

from .eda import eda_pandas_profiling

def get_dataset(
    csv_path: Path, random_state: int, test_split_ratio: float, use_eda: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path, index_col='Id')
    click.echo(f"Dataset shape: {dataset.shape}.")
    
    if use_eda:
        eda_pandas_profiling(dataset)

    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, random_state=random_state, test_size=test_split_ratio
    )
    return features_train, features_val, target_train, target_val