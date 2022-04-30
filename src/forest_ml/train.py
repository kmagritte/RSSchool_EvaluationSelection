from pathlib import Path
import click
import pandas as pd

from .data import get_dataset
from .pipeline import create_pipeline

import warnings
warnings.filterwarnings("ignore")

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",  
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-eda",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--type-scaler",
    default='StandardScaler',
    type=str,
    show_default=True,
)

def train(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_eda: bool,
    use_scaler: bool,
    type_scaler: str,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
        use_eda,
    )

    pipeline = create_pipeline(use_scaler, type_scaler, random_state)
    pipeline.fit(features_train, target_train)
    click.echo(pipeline)
