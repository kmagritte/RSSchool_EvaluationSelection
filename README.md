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
To disable automatic hyperparameter search and configure them yourself
```sh
poetry run train --hyperparameter-search=False
```
*Note!!!*  I advise you to disable automatic selection of hyperparameters. The duration of the execution is more than an hour.

You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. If you want to get an EDA report with pandas profiling, the command should look like this
```sh
poetry run train --use-eda=True
```
The report will be saved to the root of the project.

7. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
The results of my experiments:
![image](https://user-images.githubusercontent.com/98235486/166881592-4b688001-5ead-4a7d-a891-a139101d049e.png)

The following model showed itself better: *RandomForestClassifier(random_state=42, n_estimators=500, max_depth=250)* with StandardScaler and without feature engineering.

## Development

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
1. Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
To disable warnings, use the following command
```
poetry run pytest --disable-warnings
```
2. Formatting code with black
```
poetry run black src
```
```
poetry run black tests
```
Validate passed:

![image](https://user-images.githubusercontent.com/98235486/167308936-3b55f566-b885-40ef-847c-7dfa914abe5c.png)

![image](https://user-images.githubusercontent.com/98235486/167308949-2c0c0bc3-578f-41ea-884b-9e2f54ead910.png)

