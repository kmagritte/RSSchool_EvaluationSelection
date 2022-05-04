# RSSchool_EvaluationSelection
Homework for RS School Machine Learning course.

This project uses [Forest](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model for determine the forest cover types.
1. Clone this repository to your machine.
2. Download [Forest](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine.
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
Use 2 models *LogisticRegression / RandomForestClassifier*:
- *LogisticRegression(max_iter=max_iter, C=logreg_c)*
- *RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)*

For example to select a RandomForestClassifier model, run the following command with the necessary hyperparameter values. If they are not set, the default value will be used:
```sh
poetry run train --type-model=RandomForestClassifier
```
There are two types of scaler: *StandardScaler / MinMaxScaler*. If you want to remove scaler, run the command
```sh
poetry run train --use-scaler=False
```
If you want to change the scaler type, for example on MinMaxScaler
```sh
poetry run train --type-scaler=MinMaxScaler
```
Similarly for feature engineering (*PCA / LogisticRegression(solver='liblinear', penalty='l1')*). Use the default feature engineering option
```sh
poetry run train --use-feature-engineering=True
```
Change the method
```sh
poetry run train --type-feature-engineering=LogisticRegression
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
