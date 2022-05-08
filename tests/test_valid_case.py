from click.testing import CliRunner
import pytest
from sklearn.datasets._samples_generator import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os.path
import pandas as pd
from joblib import load
from src.forest_ml.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()

def test_model(
    runner: CliRunner
    ) -> None:
    with runner.isolated_filesystem():
        X, Y = make_classification(n_samples=1000, n_features=10, n_redundant=0, n_clusters_per_class=1, n_classes=3)
        df = pd.concat([pd.Series([i for i in range(0,1000)], name="Id"), pd.DataFrame(X), pd.Series(Y, name="Cover_Type")], axis=1)
        df.to_csv('for_test_valid_case.csv')

        result = runner.invoke(
            train,
            [   
                "-d",
                "for_test_valid_case.csv",
                "-s",
                "model.joblib",
                "--hyperparameter-search",
                False,
            ],
        )
        loaded_model = load("model.joblib")
        loaded_model.fit(X, Y)
        Y_pred = loaded_model['classifier'].predict(X)
        accuracy = accuracy_score(Y, Y_pred)
        precision = precision_score(Y, Y_pred, average='macro')
        recall = recall_score(Y, Y_pred, average='macro')
        f1 = f1_score(Y, Y_pred, average='macro')

        assert result.exit_code == 0
        assert os.path.exists("model.joblib")
        assert 0 <= accuracy and accuracy <= 1 
        assert 0 <= precision and precision <= 1 
        assert 0 <= recall and recall <= 1 
        assert 0 <= f1 and f1 <= 1 
        