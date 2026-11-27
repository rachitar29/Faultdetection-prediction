import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("ml_data/raw/CMAPSSData")

columns = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

for filename in ["train_FD001.txt", "test_FD001.txt"]:
    df = pd.read_csv(
        data_dir / filename,
        sep=r"\s+",
        header=None
    )
    df.columns = columns

    features = df[columns[2:]]
    values = features.to_numpy(dtype=float)

    print("\nDataset:", filename)
    print("Rows:", len(df))
    print("Engines:", df["unit"].nunique())
    print("Missing values:", int(features.isna().sum().sum()))
    print("Infinite values:", int(np.isinf(values).sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
