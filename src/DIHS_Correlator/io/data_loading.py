import pandas as pd


def load_csv(path: str, drop_unnamed_index: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if drop_unnamed_index and "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
    return df

