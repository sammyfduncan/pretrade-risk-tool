"""The contract between `build_dataset.py` and `train.py`.

These two scripts communicate through a parquet file on disk, and features are
defined by *subtraction*: everything that is not known metadata becomes a model
input. That is convenient and quietly dangerous, because the dataset is a
gitignored artifact that outlives the code which produced it. Check out a
different commit, re-run training without rebuilding, and a column can change
meaning underneath you.

The specific failure this guards against: `LABEL_COLUMNS` contains `exit_bars`
and `exit_time`, which describe how the trade *finished*. If a rename ever
leaves a stale dataset holding the old name while this list holds the new one,
the old column stops being recognised as metadata and is handed to the model as
a feature. That is direct lookahead, and it does not crash — it presents as AUC
jumping to ~0.95 and everything looking like a breakthrough. Given this project
has already been fooled several times by metrics improving for non-signal
reasons (`docs/step1_freeze_record.md`), failing loudly here is worth the lines.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as ptypes

from commodity_ml.labels import LABEL_COLUMNS
from commodity_ml.primary import SIGNAL_COLUMN

# Columns a built dataset carries for the backtest's benefit, never the model's.
METADATA_COLUMNS: list[str] = LABEL_COLUMNS + [SIGNAL_COLUMN, "close"]


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Model-input columns of a built dataset, or raise explaining why not.

    Deliberately strict: a dataset that does not match the current code is a
    reason to rebuild it, not to train on it and hope.
    """
    missing = [c for c in METADATA_COLUMNS if c not in dataset.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected metadata columns: {missing}.\n"
            "It was almost certainly built by a different version of the code, "
            "which means columns this run treats as features may be label-derived "
            "(lookahead). Rebuild it:\n"
            "    python scripts/build_dataset.py --config config/<asset>.yaml"
        )

    names = [c for c in dataset.columns if c not in METADATA_COLUMNS]
    if not names:
        raise ValueError("Dataset has no feature columns — nothing to train on.")

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
