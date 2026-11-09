import os
from typing import Protocol

import numpy as np
import pandas as pd

from app.core.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


class DataSource(Protocol):
    """Interface for sources of machine rows to replay."""

    def next_batch(self, size: int) -> pd.DataFrame:
        """Return the next ``size`` rows; fewer at the end, empty when exhausted."""
        ...

    def reset(self) -> None:
        """Restart the replay from the beginning."""
        ...

    @property
    def remaining(self) -> int:
        """Rows still available."""
        ...

    @property
    def total(self) -> int:
        """Total rows available in a full replay."""
        ...


class CsvReplaySource:
    """
    Replays a held-out test CSV as simulated batches of incoming machine data.

    Rows are shuffled once with a seeded RNG, then served without replacement
    (no machine repeats within a replay). ``reset()`` reshuffles with the same
    seed, so replays are reproducible.
    """

    def __init__(self, path: str, seed: int = 42):
        self._path = path
        self._seed = seed
        self._frame = self._load()
        self.reset()
        logger.info(
            "CSV replay source initialized: %d machines from %s", self.total, path
        )

    def _load(self) -> pd.DataFrame:
        """Load and validate the test set."""
        if not os.path.exists(self._path):
            raise FileNotFoundError(
                f"Test set not found at '{self._path}'. Expected the held-out "
                "test split of the AI4I 2020 dataset (2,000 rows, all 14 "
                "original columns). Restore or generate Data/test_set.csv "
                "(held-out Test split from Notebook/model.ipynb)."
            )
        df = pd.read_csv(self._path)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Test set at '{self._path}' is missing required columns: {missing}"
            )
        if df.empty:
            raise ValueError(f"Test set at '{self._path}' is empty")
        return df

    def reset(self) -> None:
        """Reshuffle with the same seed and restart the cursor."""
        rng = np.random.default_rng(self._seed)
        self._order = rng.permutation(len(self._frame))
        self._cursor = 0

    def next_batch(self, size: int) -> pd.DataFrame:
        """Return the next ``size`` rows (sampling without replacement)."""
        remaining_rows = max(len(self._order) - self._cursor, 0)
        take = min(size, remaining_rows)
        if take <= 0:
            return self._frame.iloc[0:0].copy()
        idx = self._order[self._cursor : self._cursor + take]
        self._cursor += take
        return self._frame.iloc[idx].copy()

    @property
    def remaining(self) -> int:
        """Rows still available."""
        return max(len(self._order) - self._cursor, 0)

    @property
    def total(self) -> int:
        """Total rows available in a full replay."""
        return len(self._order)
