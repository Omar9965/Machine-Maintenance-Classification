"""Tests for the CsvReplaySource — determinism, coverage, exhaustion, reset."""

import pytest

from app.data_sources.csv_replay import CsvReplaySource


@pytest.fixture()
def source(settings):
    return CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)


def test_same_seed_deterministic(settings, test_set_df):
    a = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    b = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    batch_a = a.next_batch(10)
    batch_b = b.next_batch(10)
    assert batch_a["UDI"].tolist() == batch_b["UDI"].tolist()


def test_total_and_remaining(settings, test_set_df):
    source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    assert source.total == len(test_set_df)
    assert source.remaining == source.total
    source.next_batch(7)
    assert source.remaining == source.total - 7


def test_no_duplicate_udi_across_batches(settings):
    source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    seen = set()
    while True:
        batch = source.next_batch(64)
        if batch.empty:
            break
        for udi in batch["UDI"]:
            assert udi not in seen, f"UDI {udi} repeated within a replay"
            seen.add(int(udi))


def test_batches_sum_to_total(settings, test_set_df):
    source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    total_rows = 0
    while True:
        batch = source.next_batch(137)
        if batch.empty:
            break
        total_rows += len(batch)
    assert total_rows == len(test_set_df)


def test_partial_last_batch(settings):
    source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    # 2000 % 300 == 200 → last batch is a partial one
    sizes = []
    while True:
        batch = source.next_batch(300)
        if batch.empty:
            break
        sizes.append(len(batch))
    assert sizes[-1] == 200
    assert all(size == 300 for size in sizes[:-1])


def test_empty_after_exhaustion(settings):
    source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    while source.remaining > 0:
        source.next_batch(500)
    batch = source.next_batch(10)
    assert batch.empty
    assert source.remaining == 0


def test_reset_restarts_same_order(settings):
    source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    first_run = []
    while True:
        batch = source.next_batch(250)
        if batch.empty:
            break
        first_run.extend(batch["UDI"].tolist())

    source.reset()
    second_run = []
    while True:
        batch = source.next_batch(250)
        if batch.empty:
            break
        second_run.extend(batch["UDI"].tolist())

    assert first_run == second_run


def test_returns_copies_not_views(settings):
    source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
    batch = source.next_batch(5)
    batch.iloc[0, batch.columns.get_loc("Torque [Nm]")] = -999
    fresh = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED).next_batch(5)
    assert fresh.iloc[0]["Torque [Nm]"] != -999


def test_missing_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CsvReplaySource(str(tmp_path / "does_not_exist.csv"), 42)
