from pathlib import Path
from joblib import dump
import click
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

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
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--n-splits",
    default=5,
    type=int,
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
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
)

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    n_splits: int,
    use_eda: bool,
    use_scaler: bool,
    type_scaler: str,
    type_model: str,
    max_iter: int,
    logreg_c: float,
    n_estimators: int,
    max_depth: int,
) -> None:
    features, target = get_dataset(
        dataset_path,
        use_eda,
    )
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    cv_outer = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipeline = create_pipeline(use_scaler, type_scaler, type_model, random_state, max_iter, logreg_c, n_estimators, max_depth)
    scores = cross_validate(pipeline, features, target, scoring=scoring, cv=cv_outer)
    click.echo(pipeline)
    
    click.echo(f"Accuracy: {sum(scores['test_accuracy'])/n_splits},")
    click.echo(f"F1 score: {sum(scores['test_f1_macro'])/n_splits},")
    click.echo(f"Precision score: {sum(scores['test_precision_macro'])/n_splits},")
    click.echo(f"Recall score: {sum(scores['test_recall_macro'])/n_splits}.")
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")

