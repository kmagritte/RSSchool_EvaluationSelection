from pathlib import Path
import click
import pandas as pd

from .data import get_dataset

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",  
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
def train(
    dataset_path: Path
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
    )
