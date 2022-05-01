import click
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def create_pipeline(
    use_scaler: bool, type_scaler: str, type_model: str, random_state: int, max_iter: int,
    logreg_c: float,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler and type_scaler.lower() == 'standardscaler':
        pipeline_steps.append(("scaler", StandardScaler()))
    elif use_scaler and type_scaler.lower() == 'minmaxscaler':
        pipeline_steps.append(("scaler", MinMaxScaler()))
    else:
        click.echo('Error: Invalid scaler. The default option will be used.')
        pipeline_steps.append(("scaler", StandardScaler()))

    if type_model.lower() == 'logisticregression':
        pipeline_steps.append(("classifier", LogisticRegression(random_state=random_state, max_iter=max_iter, C=logreg_c,)))
    elif type_model.lower() == 'randomforestclassifier':
        pipeline_steps.append(("classifier", RandomForestClassifier()))
    else:
        click.echo('Error: Invalid classifier. The default option will be used.')
        pipeline_steps.append(("classifier", LogisticRegression(random_state=random_state, max_iter=max_iter, C=logreg_c,)))

    return Pipeline(steps=pipeline_steps)
