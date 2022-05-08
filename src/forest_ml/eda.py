import pandas as pd
from pandas_profiling import ProfileReport


def eda_pandas_profiling(
    dataset: pd.DataFrame,
) -> None:
    TITLE = "Pandas Profiling Report"
    profile = ProfileReport(dataset, title=TITLE, explorative=True)
    profile.to_file("Report.html")
