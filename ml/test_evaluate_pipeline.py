import pandas as pd

from ml import evaluate 
from ml.utils import resample_balance
import pytest


def _make_sample_df():
    return pd.DataFrame({
        "text": ["real one", "fake one", "real two", "fake two", "real three"],
        "label": ["real", "fake", "real", "fake", "real"],
        "num": [1,2,3,2,1]
    })

def test_load_test_data_detects_columns(tmp_path):
    p = tmp_path / "test.csv"
    df = _make_sample_df()
    df.to_csv(p, index=False)
    loaded_df, text_col, label_col = evaluate.load_test_data(str(p), has_header=True)
    assert text_col in loaded_df.columns or isinstance(text_col, int)
    assert label_col in loaded_df.columns or isinstance(label_col, int)
    assert len(loaded_df) == 5

def test_resample_upsample_and_downsample():
    df = _make_sample_df()
    up = resample_balance(df, label_col='label', method='upsample', random_state=0)
    counts_up = up['label'].value_counts()
    assert counts_up.iloc[0] == counts_up.iloc[1] 
    down = resample_balance(df, label_col='label', method='downsamople', random_state=0)
    counts_down = down['label'].value_counts()
    assert counts_down.iloc[0] == counts_down.iloc[1]