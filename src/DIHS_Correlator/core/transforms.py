import numpy as np
import pandas as pd
from skbio.stats.composition import clr, ilr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


BASE_TRANSFORMATIONS = {
    0: "none",
    1: "ilr",
    2: "clr",
    3: "scaled",
}


def set_feature_columns(data: pd.DataFrame, exclude=()):
    cols = [c for c in data.columns if c not in exclude]
    cols = [c for c in cols if np.issubdtype(data[c].dtype, np.number)]
    if len(cols) == 0:
        raise ValueError("No valid feature columns found in the data.")
    return cols


def apply_transformation(
    data: pd.DataFrame,
    feature_columns,
    transform_type: int,
    class_column: str = "controlcode",
):
    if transform_type == 0:
        features = feature_columns
        transformed_data = data[features].values

    elif transform_type == 1:
        features = [f"ILR_{i+1}" for i in range(len(feature_columns) - 1)]
        x = data[feature_columns].to_numpy(dtype=np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        x = x + 1e-10
        row_sums = x.sum(axis=1, keepdims=True)
        x = x / row_sums
        transformed_data = ilr(x)
        scaler = StandardScaler(with_mean=True, with_std=True)
        transformed_data = scaler.fit_transform(transformed_data)

    elif transform_type == 2:
        features = feature_columns
        x = data[feature_columns].to_numpy(dtype=np.float64)
        x = np.where(np.isfinite(x), x, 0.0)
        x = x + 1e-10
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        x = x / row_sums
        transformed_data = clr(x)
        scaler = StandardScaler(with_mean=True, with_std=True)
        transformed_data = scaler.fit_transform(transformed_data)

    elif transform_type == 3:
        features = feature_columns
        x = data[feature_columns].to_numpy(dtype=np.float64)
        x = np.where(np.isinf(x), np.nan, x)
        imputer = SimpleImputer(strategy="median")
        x_imp = imputer.fit_transform(x)
        scaler = StandardScaler(with_mean=True, with_std=True)
        transformed_data = scaler.fit_transform(x_imp)

    else:
        raise ValueError(f"Transformation type '{transform_type}' is not supported.")

    if np.any(np.isnan(transformed_data)) or np.any(np.isinf(transformed_data)):
        transformed_data = np.where(
            np.isinf(transformed_data),
            np.sign(transformed_data) * 1e10,
            transformed_data,
        )
        transformed_data = np.where(np.isnan(transformed_data), 0.0, transformed_data)

    transformed_df = pd.DataFrame(transformed_data, columns=features, index=data.index)
    transformed_df[class_column] = data[class_column].values
    return transformed_df, features
