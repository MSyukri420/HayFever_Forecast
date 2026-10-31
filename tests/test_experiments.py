"""Keep-rule tests for the experiment log (no data needed)."""

import pytest

from src import experiments as E


@pytest.fixture(autouse=True)
def temp_log(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "LOG_PATH", tmp_path / "experiments.csv")


REF = {2021: 4.0, 2022: 4.0, 2023: 4.0}


def test_first_run_is_kept_as_reference():
    assert E.log_run("antihist", "lgb_poisson", "reference", REF)
    assert E.best_run("antihist")["val_mean"] == 4.0


def test_kept_only_if_better_every_year_and_by_min_gain():
    E.log_run("antihist", "lgb_poisson", "reference", REF)
    big = 4.0 * (1 - E.MIN_GAIN) - 0.05
    assert E.is_improvement("antihist", {2021: big, 2022: big, 2023: big})
    # a large mean gain driven by one year, while another year gets worse
    assert not E.is_improvement("antihist", {2021: 2.0, 2022: 4.1, 2023: 3.9})
    # better every year, but by less than MIN_GAIN
    small = 4.0 * (1 - E.MIN_GAIN / 2)
    assert not E.is_improvement("antihist", {2021: small, 2022: small, 2023: small})


def test_rejected_runs_do_not_become_the_reference():
    E.log_run("antihist", "lgb_poisson", "reference", REF)
    assert not E.log_run("antihist", "lgb_poisson", "worse", {2021: 5.0, 2022: 5.0, 2023: 5.0})
    assert E.best_run("antihist")["change"] == "reference"
    assert E.best_run("nasal") is None
