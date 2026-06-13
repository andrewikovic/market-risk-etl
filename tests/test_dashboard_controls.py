from datetime import date

import pandas as pd

from dashboards import common


def test_realized_portfolio_return_series_matches_pipeline_metric_input():
    portfolio_values = pd.DataFrame(
        {
            "value_date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-04"]),
            "daily_return": [0.02, 0.0, 0.01, float("inf")],
        }
    )

    returns = common.realized_portfolio_return_series(portfolio_values)

    assert returns.tolist() == [0.01, 0.02]
    assert returns.index.tolist() == pd.to_datetime(["2024-01-02", "2024-01-03"]).tolist()


def test_data_source_control_persists_selection_across_page_renders(monkeypatch):
    session_state = {}
    first_render = _FakeStreamlit(data_source="live", session_state=session_state)
    monkeypatch.setattr(common, "st", first_render)

    assert common.render_data_source_control() == "live"

    second_render = _FakeStreamlit(session_state=session_state)
    monkeypatch.setattr(common, "st", second_render)

    assert common.render_data_source_control() == "live"


def test_price_history_controls_are_hidden_for_non_live_sources():
    assert common.render_price_history_controls("sample") == {
        "price_start": None,
        "price_end": None,
        "price_lookback_days": None,
        "price_period": None,
    }


def test_period_history_control_returns_yfinance_period(monkeypatch):
    monkeypatch.setattr(common, "st", _FakeStreamlit(history_mode="period", period="5y"))

    assert common.render_price_history_controls("live") == {
        "price_start": None,
        "price_end": None,
        "price_lookback_days": None,
        "price_period": "5y",
    }


def test_date_range_history_control_returns_iso_dates(monkeypatch):
    monkeypatch.setattr(
        common,
        "st",
        _FakeStreamlit(history_mode="date_range", start_date=date(2020, 1, 1), end_date=date(2021, 1, 1)),
    )

    assert common.render_price_history_controls("live") == {
        "price_start": "2020-01-01",
        "price_end": "2021-01-01",
        "price_lookback_days": None,
        "price_period": None,
    }


def test_lookback_history_control_returns_day_count(monkeypatch):
    monkeypatch.setattr(common, "st", _FakeStreamlit(history_mode="lookback", lookback_days=504))

    assert common.render_price_history_controls("live") == {
        "price_start": None,
        "price_end": None,
        "price_lookback_days": 504,
        "price_period": None,
    }


class _FakeStreamlit:
    def __init__(
        self,
        data_source: str | None = None,
        history_mode: str = "full",
        period: str = "max",
        start_date: date | None = None,
        end_date: date | None = None,
        lookback_days: int = 756,
        session_state: dict | None = None,
    ):
        self.session_state = session_state if session_state is not None else {}
        self.sidebar = _FakeSidebar(data_source, history_mode, period, start_date, end_date, lookback_days)

    def stop(self):
        raise RuntimeError("st.stop called")


class _FakeSidebar:
    def __init__(
        self,
        data_source: str | None,
        history_mode: str,
        period: str,
        start_date: date | None,
        end_date: date | None,
        lookback_days: int,
    ):
        self.data_source = data_source
        self.history_mode = history_mode
        self.period = period
        self.start_date = start_date or date(2020, 1, 1)
        self.end_date = end_date or date(2021, 1, 1)
        self.lookback_days = lookback_days

    def radio(self, label, options, *_args, index=0, **_kwargs):
        if label == "Data source":
            return self.data_source or options[index]
        if label == "Yahoo history":
            return self.history_mode or options[index]
        raise AssertionError(f"Unexpected radio label: {label}")

    def selectbox(self, label, *_args, **_kwargs):
        if label == "Period":
            return self.period
        raise AssertionError(f"Unexpected selectbox label: {label}")

    def date_input(self, label, *_args, **_kwargs):
        if label == "Start date":
            return self.start_date
        if label == "End date":
            return self.end_date
        raise AssertionError(f"Unexpected date input label: {label}")

    def number_input(self, label, *_args, **_kwargs):
        if label == "Lookback days":
            return self.lookback_days
        raise AssertionError(f"Unexpected number input label: {label}")

    def caption(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None
