import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    data = df.copy()
    data = data.drop_duplicates()
    if "Year" in data.columns:
        data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
        # Correct invalid-year filtering: keep plausible years, mark outliers missing.
        data.loc[~data["Year"].between(1950, 2035), "Year"] = np.nan
    data["Global_Sales"] = pd.to_numeric(data["Global_Sales"], errors="coerce")
    data = data.dropna(subset=["Global_Sales"])
    data = data[data["Global_Sales"] >= 0]
    for col in ["Platform", "Genre", "Publisher"]:
        if col not in data.columns:
            raise ValueError(f"Required column missing: {col}")
    data = data.dropna(subset=["Platform", "Genre", "Publisher"])
    return data.reset_index(drop=True)

def dataset_profile(raw, cleaned):
    steps = [
        ("Original rows", len(raw)),
        ("Duplicate rows removed", int(raw.duplicated().sum())),
        ("Rows missing Global_Sales", int(raw["Global_Sales"].isna().sum())),
        ("Negative Global_Sales rows", int((pd.to_numeric(raw["Global_Sales"], errors="coerce") < 0).sum())),
        ("Rows removed in total", len(raw)-len(cleaned)),
        ("Rows after cleaning", len(cleaned)),
    ]
    return {"cleaning": pd.DataFrame(steps, columns=["Cleaning operation","Records"])}
