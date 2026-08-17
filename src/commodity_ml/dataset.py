"""Contract between ``build_dataset.py`` and ``train.py``.

Those scripts communicate through a Parquet file on disk. Features are defined
by subtraction: every column that is not known metadata becomes model input.

LABEL_COLUMNS contains exit_bars, exit_time. 
If a rename leaves a stale dataset holding the old name while this list holds
the new one, the old column stops being recognised and is handed to the model
as a feature.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from commodity_ml.labels import LABEL_COLUMNS
from commodity_ml.primary import SIGNAL_COLUMN

# Columns a built dataset carries for the backtest's benefit, never the model's.
METADATA_COLUMNS: list[str] = LABEL_COLUMNS + [SIGNAL_COLUMN, "close"]


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Model input cols of a built dataset, or raise explaining why not.

    Dataset not matching current code causes rebuild, instead of training on it.
    """
    missing = [c for c in METADATA_COLUMNS if c not in dataset.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected metadata columns: {missing}.\n"
            "It was likely built by a different version of the code, meaning "
            "columns this run treats as features may be label-derived. "
            "Rebuild:\n"
            "    python scripts/build_dataset.py --config config/<asset>.yaml"
        )

    names = [c for c in dataset.columns if c not in METADATA_COLUMNS]
    if not names:
        raise ValueError("Dataset contains no feature columns, nothing to train on")

    # exit_time is tz-aware datetime and label/exit_bars are integral, so a
    # non-numeric column here is a strong signal that metadata has leaked in
    # under an unrecognised name.
    non_numeric = [c for c in names if not ptypes.is_numeric_dtype(dataset[c])]
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns would be passed to the model as features: "
            f"{non_numeric}. Features must be numeric; this usually means a "
            "metadata column is not being recognised."
        )
    return names
