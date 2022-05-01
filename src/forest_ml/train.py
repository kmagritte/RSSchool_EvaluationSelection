from pathlib import Path
import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

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
@click.option(
    "--type-model",
    default='LogisticRegression',
    type=str,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)

def train(
    dataset_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_eda: bool,
    use_scaler: bool,
    type_scaler: str,
    type_model: str,
    max_iter: int,
    logreg_c: float,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
        use_eda,
    )

    pipeline = create_pipeline(use_scaler, type_scaler, type_model, random_state, max_iter, logreg_c)
    pipeline.fit(features_train, target_train)
    click.echo(pipeline)

    accuracy = accuracy_score(target_val, pipeline.predict(features_val))
    f1 = f1_score(target_val, pipeline.predict(features_val), average='macro') 
    kappa_score = cohen_kappa_score(target_val, pipeline.predict(features_val))
    click.echo(f"Accuracy: {accuracy},")
    click.echo(f"F1 score: {f1},")
    click.echo(f"Kappa score: {kappa_score}.")

