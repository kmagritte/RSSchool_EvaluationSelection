import click
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_pipeline(
    use_scaler: bool, type_scaler: str, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler and type_scaler.lower() == 'standardscaler':
        pipeline_steps.append(("scaler", StandardScaler()))
    elif use_scaler and type_scaler.lower() == 'minmaxscaler':
        pipeline_steps.append(("scaler", MinMaxScaler()))
    else:
        click.echo('Error: Invalid scaler. The default option will be used.')
        pipeline_steps.append(("scaler", StandardScaler()))
        
    return Pipeline(steps=pipeline_steps)
