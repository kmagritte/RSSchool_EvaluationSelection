from pathlib import Path
from joblib import dump
import click
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
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
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
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
    default="StandardScaler",
    type=str,
    show_default=True,
)
@click.option(
    "--use-feature-engineering",
    default=False,
    type=bool,
    show_default=True,
)
@click.option(
    "--type-feature-engineering",
    default="PCA",
    type=str,
    show_default=True,
)
@click.option(
    "--type-model",
    default="LogisticRegression",
    type=str,
    show_default=True,
)
@click.option(
    "--hyperparameter-search",
    default=True,
    type=bool,
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
    use_feature_engineering: bool,
    type_feature_engineering: str,
    hyperparameter_search: bool,
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
    with mlflow.start_run():
        scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        cv_outer = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        pipeline = create_pipeline(
            use_scaler,
            type_scaler,
            use_feature_engineering,
            type_feature_engineering,
            type_model,
            random_state,
            max_iter,
            logreg_c,
            n_estimators,
            max_depth,
            hyperparameter_search,
        )
        scores = cross_validate(
            pipeline, features, target, scoring=scoring, cv=cv_outer
        )
        click.echo(pipeline)
        click.echo(f"Accuracy: {np.mean(scores['test_accuracy'])},")
        click.echo(f"F1 score: {np.mean(scores['test_f1_macro'])},")
        click.echo(f"Precision score: {np.mean(scores['test_precision_macro'])},")
        click.echo(f"Recall score: {np.mean(scores['test_recall_macro'])}.")

        if hyperparameter_search:
            mlflow.log_param("hyperparameter_search", hyperparameter_search)
        else:
            mlflow.log_param("hyperparameter_search", hyperparameter_search)
            if type_model.lower() == "randomforestclassifier":
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("max_iter", None)
                mlflow.log_param("logreg_c", None)

            else:
                mlflow.log_param("n_estimators", None)
                mlflow.log_param("max_depth", None)
                mlflow.log_param("max_iter", max_iter)
                mlflow.log_param("logreg_c", logreg_c)

        mlflow.log_param("use_scaler", use_scaler)
        if use_scaler:
            mlflow.log_param("type_scaler", pipeline["scaler"])
        else:
            mlflow.log_param("type_scaler", None)

        mlflow.log_param("use_feature_engineering", use_feature_engineering)
        if use_feature_engineering:
            mlflow.log_param(
                "type_feature_engineering", pipeline["feature_engineering"]
            )
        else:
            mlflow.log_param("type_feature_engineering", None)

        mlflow.log_metric("Accuracy", np.mean(scores["test_accuracy"]))
        mlflow.log_metric("F1", np.mean(scores["test_f1_macro"]))
        mlflow.log_metric("Precision", np.mean(scores["test_precision_macro"]))
        mlflow.log_metric("Recall", np.mean(scores["test_recall_macro"]))

        mlflow.sklearn.log_model(pipeline, "model")

        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
