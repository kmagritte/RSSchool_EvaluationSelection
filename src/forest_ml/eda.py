import pandas as pd
from pandas_profiling import ProfileReport

def eda_pandas_profiling(
    dataset: pd.DataFrame,
) -> None:
    profile = ProfileReport(dataset, title="Pandas Profiling Report", explorative=True)
    profile.to_file("Report.html")
