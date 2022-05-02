from pathlib import Path
from typing import Tuple
import click
import pandas as pd

from .eda import eda_pandas_profiling

def get_dataset(
    csv_path: Path, use_eda: bool,
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path, index_col='Id')
    click.echo(f"Dataset shape: {dataset.shape}.")
    
    if use_eda:
        eda_pandas_profiling(dataset)

    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]

    return features, target