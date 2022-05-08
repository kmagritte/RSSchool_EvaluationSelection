from click.testing import CliRunner
import pytest

from src.forest_ml.train import train

@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_dataset_path(
    runner: CliRunner
) -> None:
    """It fails when dataset_path is not exist."""
    result = runner.invoke(
        train,
        [   
            "-d",
            "d/data/train.csv",
        ],

    )
    assert result.exit_code == 2
    assert "Invalid value for '-d' / '--dataset-path'" in result.output
