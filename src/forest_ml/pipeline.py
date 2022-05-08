import click
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, GridSearchCV


def create_pipeline(
    use_scaler: bool,
    type_scaler: str,
    use_feature_engineering: bool,
    type_feature_engineering: str,
    type_model: str,
    random_state: int,
    max_iter: int,
    logreg_c: float,
    n_estimators: int,
    max_depth: int,
    hyperparameter_search: bool,
) -> Pipeline:
    pipeline_steps = []

    param_rf = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 100, 200, 300, 400, 500],
        "criterion": ["gini", "entropy"],
    }

    param_lr = {
        "max_iter": [100, 200, 300, 400, 500],
        "C": [0.1, 0.5, 1.0, 1.2, 1.5, 2],
    }
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)

    if use_scaler and type_scaler.lower() == "standardscaler":
        pipeline_steps.append(("scaler", StandardScaler()))
    elif use_scaler and type_scaler.lower() == "minmaxscaler":
        pipeline_steps.append(("scaler", MinMaxScaler()))
    elif use_scaler:
        click.echo("Error: Invalid scaler. The default option will be used.")
        pipeline_steps.append(("scaler", StandardScaler()))

    if use_feature_engineering and type_feature_engineering.lower() == "pca":
        pipeline_steps.append(
            ("feature_engineering", PCA(random_state=random_state, n_components=15))
        )
    elif (
        use_feature_engineering
        and type_feature_engineering.lower() == "logisticregression"
    ):
        pipeline_steps.append(
            (
                "feature_engineering",
                SelectFromModel(
                    LogisticRegression(
                        random_state=random_state, solver="liblinear", penalty="l1"
                    )
                ),
            )
        )
    elif use_feature_engineering:
        click.echo(
            "Error: Invalid feature engineering method. The default option will be used."
        )
        pipeline_steps.append(
            ("feature_engineering", PCA(random_state=random_state, n_components=15))
        )

    if hyperparameter_search:
        if type_model.lower() == "logisticregression":
            search = GridSearchCV(
                LogisticRegression(random_state=random_state),
                param_lr,
                scoring="accuracy",
                cv=cv_inner,
                refit=True,
            )
            pipeline_steps.append(("classifier", search))
        elif type_model.lower() == "randomforestclassifier":
            search = GridSearchCV(
                RandomForestClassifier(random_state=random_state),
                param_rf,
                scoring="accuracy",
                cv=cv_inner,
                refit=True,
            )
            pipeline_steps.append(("classifier", search))
        else:
            click.echo("Error: Invalid classifier. The default option will be used.")
            search = GridSearchCV(
                LogisticRegression(random_state=random_state),
                param_lr,
                scoring="accuracy",
                cv=cv_inner,
                refit=True,
            )
            pipeline_steps.append(("classifier", search))

    else:
        if type_model.lower() == "logisticregression":
            pipeline_steps.append(
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_state,
                        max_iter=max_iter,
                        C=logreg_c,
                    ),
                )
            )
        elif type_model.lower() == "randomforestclassifier":
            pipeline_steps.append(
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=random_state,
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                    ),
                )
            )
        else:
            click.echo("Error: Invalid classifier. The default option will be used.")
            pipeline_steps.append(
                (
                    "classifier",
                    LogisticRegression(
                        random_state=random_state,
                        max_iter=max_iter,
                        C=logreg_c,
                    ),
                )
            )

    return Pipeline(steps=pipeline_steps)
