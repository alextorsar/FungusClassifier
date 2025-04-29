import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional

__all__ = [
    "ColumnDropper",
    "SpectrumExpander",
    "IntensityScaler",
    "TICLogTransformer",
]


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Transformer that drops specified columns from a pandas DataFrame.

    Parameters
    ----------
    columns : List[str]
        List of column names to drop.
    """
    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnDropper":
        # No fitting necessary
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Drop columns, ignore errors if not present
        return X.drop(columns=self.columns, errors='ignore')


class SpectrumExpander(BaseEstimator, TransformerMixin):
    """
    Transformer that expands a column of list/array spectra into separate numeric columns.

    Parameters
    ----------
    spectrum_col : str, default="spectrum"
        Name of the column containing list or array spectra.
    """
    def __init__(self, spectrum_col: str = "spectrum"):
        self.spectrum_col = spectrum_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SpectrumExpander":
        # No fitting necessary
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Copy to avoid modifying original
        df = X.copy()
        # Convert spectrum column to DataFrame of numeric columns
        spec_data = df[self.spectrum_col].tolist()
        spec_df = pd.DataFrame(
            spec_data,
            index=df.index,
            columns=[f"{self.spectrum_col}_{i}" for i in range(len(spec_data[0]))]
        )
        # Concatenate and drop original column
        df = pd.concat([df.drop(columns=[self.spectrum_col]), spec_df], axis=1)
        return df


class IntensityScaler(BaseEstimator, TransformerMixin):
    """
    Transformer that scales numeric values by a constant factor.

    Parameters
    ----------
    factor : float, default=1e4
        Multiplicative factor to apply to all values.
    """
    def __init__(self, factor: float = 1e4):
        self.factor = factor

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "IntensityScaler":
        # No fitting necessary
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Multiply all values by the factor
        return X * self.factor


class TICLogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that performs total ion current (TIC) normalization row-wise and applies log1p.

    Parameters
    ----------
    eps : float, default=1e-8
        Small constant to add to row sums for numerical stability.
    """
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TICLogTransformer":
        # No fitting necessary
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Row-wise TIC normalization
        tic = X.div(X.sum(axis=1) + self.eps, axis=0)
        # log1p transform
        out = np.log1p(tic)
        # Preserve index and columns
        out.index, out.columns = X.index, X.columns
        return out
