import math

import numpy as np
import pandas as pd

from src.risk.backtesting import generate_exception_report, kupiec_pof_test


def test_exception_report_tracks_breaches_and_kupiec_summary():
    dates = pd.date_range("2024-01-01", periods=10)
    realized_pnl = pd.Series([-50, -120, 25, -80, -200, 10, 30, -40, -70, 5], index=dates)

    report = generate_exception_report(realized_pnl, 100.0, confidence_level=0.95)

    assert report["breach"].sum() == 2
    assert report.loc[report["breach"], "breach_severity"].tolist() == [20.0, 100.0]
    assert report["total_observations"].iloc[0] == 10
    assert report["number_of_exceptions"].iloc[0] == 2
    assert np.isclose(report["expected_exceptions"].iloc[0], 0.5)
    assert np.isclose(report["exception_ratio"].iloc[0], 4.0)
    assert report["pass_fail"].iloc[0] in {"pass", "fail"}


def test_kupiec_pof_supports_configurable_confidence_level():
    result = kupiec_pof_test(exceptions=1, observations=40, confidence_level=0.975)
    expected_probability = 0.025
    observed_probability = 1 / 40
    log_null = math.log(expected_probability) + 39 * math.log(1 - expected_probability)
    log_alt = math.log(observed_probability) + 39 * math.log(1 - observed_probability)
    expected_statistic = -2 * (log_null - log_alt)

    assert np.isclose(result["expected_exceptions"], 1.0)
    assert np.isclose(result["kupiec_statistic"], expected_statistic)
    assert np.isclose(result["p_value"], math.erfc(math.sqrt(expected_statistic / 2.0)))
