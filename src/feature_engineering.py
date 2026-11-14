import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def add_engineered_features(df):
    df = df.copy()
    cols = set(df.columns)
    if {"Total Fwd Packets", "Total Backward Packets"} <= cols:
        df["Packet Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"] + 1.0)
    if {"Total Length of Fwd Packets", "Total Length of Bwd Packets", "Total Fwd Packets", "Total Backward Packets"} <= cols:
        total_bytes = df["Total Length of Fwd Packets"] + df["Total Length of Bwd Packets"]
        total_pkts = df["Total Fwd Packets"] + df["Total Backward Packets"]
        df["Bytes Per Packet"] = total_bytes / (total_pkts + 1.0)
    return df.replace([np.inf, -np.inf], np.nan)


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def build_preprocessor(scale=True):
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
             ("clipper", OutlierClipper())]
    if scale:
        steps.append(("scaler", MinMaxScaler()))
    return Pipeline(steps)
