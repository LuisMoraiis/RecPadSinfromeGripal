import pandas as pd

def selectFeatures(df: pd.DataFrame, feature: str) -> list[str]:
    varIndependentes = []
    for col in df.columns:
        if col.startswith(feature):
            varIndependentes.append(col)

    return varIndependentes
