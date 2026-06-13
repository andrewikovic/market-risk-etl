from datetime import UTC, datetime

import pandas as pd

from dashboards import common


def test_render_table_download_serializes_csv_with_slugged_filename(monkeypatch):
    fake_streamlit = _FakeDownloadStreamlit()
    monkeypatch.setattr(common, "st", fake_streamlit)

    common.render_table_download(pd.DataFrame({"ticker": ["AAPL"], "value": [100.0]}), "VaR & ES")

    call = fake_streamlit.download_calls[0]
    assert call["file_name"] == "var_es.csv"
    assert call["mime"] == "text/csv"
    assert call["data"] == b"ticker,value\nAAPL,100.0\n"
    assert call["disabled"] is False


def test_render_risk_pack_downloads_creates_csv_html_and_pdf_buttons(monkeypatch):
    fake_streamlit = _FakeRiskPackStreamlit()
    monkeypatch.setattr(common, "st", fake_streamlit)

    pack = common.render_risk_pack_downloads(
        _sample_dashboard_data(),
        stress_results=_sample_stress_results(),
        key_prefix="test_pack",
    )

    assert pack.sections
    assert fake_streamlit.subheaders == ["Risk Pack Exports"]
    assert [(call["label"], call["mime"], call["file_name"].split(".")[-1]) for call in fake_streamlit.download_calls] == [
        ("Download risk-pack CSV", "text/csv", "csv"),
        ("Download risk-pack HTML", "text/html", "html"),
        ("Download risk-pack PDF", "application/pdf", "pdf"),
    ]
    assert [call["key"] for call in fake_streamlit.download_calls] == [
        "test_pack_csv",
        "test_pack_html",
        "test_pack_pdf",
    ]


def test_build_risk_pack_creates_required_sections():
    pack = common.build_risk_pack(
        _sample_dashboard_data(),
        stress_results=_sample_stress_results(),
        generated_at=datetime(2024, 2, 6, 12, 0, tzinfo=UTC),
    )
    sections = dict(pack.sections)

    assert list(sections) == ["Portfolio Summary", "VaR/ES", "Stress Tests", "Exposures", "Backtesting"]
    assert sections["Portfolio Summary"].loc[0, "holdings"] == 2
    assert sections["VaR/ES"]["metric"].tolist() == ["Historical VaR", "Parametric VaR", "Expected Shortfall"]
    assert sections["Stress Tests"].loc[0, "scenario_name"] == "Equity Shock"
    assert sections["Exposures"]["market_value"].tolist() == [125.0, 75.0]
    assert sections["Backtesting"].loc[0, "number_of_exceptions"] == 2


def test_build_risk_pack_uses_supplied_stress_results_without_recomputing(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_stress_test should not be called")

    monkeypatch.setattr(common, "run_stress_test", fail_if_called)

    pack = common.build_risk_pack(
        _sample_dashboard_data(),
        stress_results=_sample_stress_results(),
        generated_at=datetime(2024, 2, 6, 12, 0, tzinfo=UTC),
    )

    assert dict(pack.sections)["Stress Tests"].loc[0, "scenario_name"] == "Equity Shock"


def test_risk_pack_missing_data_still_serializes_with_status_rows():
    pack = common.build_risk_pack({}, scenarios=[], generated_at=datetime(2024, 2, 6, 12, 0, tzinfo=UTC))
    sections = dict(pack.sections)

    assert sections["Portfolio Summary"].loc[0, "status"] == "No portfolio value rows are available."
    assert sections["VaR/ES"].loc[0, "status"] == "No VaR or expected shortfall metrics are available."
    assert sections["Stress Tests"].loc[0, "status"] == "No stress test results are available."
    assert sections["Exposures"].loc[0, "status"] == "No exposure rows are available."
    assert sections["Backtesting"].loc[0, "status"] == "No VaR backtesting rows are available."
    assert common.risk_pack_to_csv_bytes(pack).startswith(b"Market Risk Pack")
    assert b"No exposure rows are available." in common.risk_pack_to_html_bytes(pack)
    assert common.risk_pack_to_pdf_bytes(pack).startswith(b"%PDF-1.4")


def test_risk_pack_serializers_emit_csv_html_and_pdf():
    pack = common.build_risk_pack(
        _sample_dashboard_data(),
        stress_results=_sample_stress_results(),
        generated_at=datetime(2024, 2, 6, 12, 0, tzinfo=UTC),
    )

    csv_text = common.risk_pack_to_csv_bytes(pack).decode("utf-8")
    html_text = common.risk_pack_to_html_bytes(pack).decode("utf-8")
    pdf_bytes = common.risk_pack_to_pdf_bytes(pack)

    assert "Portfolio Summary" in csv_text
    assert "Stress Tests" in csv_text
    assert "<h2>VaR/ES</h2>" in html_text
    assert "<td>Equity Shock</td>" in html_text
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")


def _sample_dashboard_data():
    return {
        "portfolio_values": pd.DataFrame(
            {
                "portfolio_name": ["P", "P"],
                "value_date": pd.to_datetime(["2024-02-05", "2024-02-06"]),
                "market_value": [190.0, 200.0],
                "daily_pnl": [0.0, 10.0],
                "daily_return": [0.0, 0.05],
                "cumulative_return": [0.0, 0.05],
            }
        ),
        "current_positions": pd.DataFrame(
            {
                "portfolio_name": ["P", "P"],
                "ticker": ["AAPL", "TLT"],
                "asset_class": ["Equity", "Fixed Income"],
                "sector": ["Technology", "Treasury"],
                "currency": ["USD", "USD"],
                "position_value": [125.0, 75.0],
            }
        ),
        "risk_metrics": {
            "historical_var_95": 12.0,
            "parametric_var_95": 11.5,
            "expected_shortfall_95": 14.0,
        },
        "exposures": pd.DataFrame(
            {
                "exposure_type": ["sector", "sector"],
                "exposure_name": ["Technology", "Treasury"],
                "market_value": [125.0, 75.0],
                "weight": [0.625, 0.375],
            }
        ),
        "var_backtest": pd.DataFrame(
            {
                "portfolio_name": ["P", "P"],
                "date": pd.to_datetime(["2024-02-05", "2024-02-06"]),
                "confidence_level": [0.95, 0.95],
                "total_observations": [20, 21],
                "number_of_exceptions": [1, 2],
                "expected_exceptions": [1.0, 1.1],
                "exception_ratio": [1.0, 1.9],
                "kupiec_statistic": [0.1, 0.2],
                "p_value": [0.9, 0.8],
                "pass_fail": ["pass", "pass"],
            }
        ),
    }


def _sample_stress_results():
    return {
        "scenario_results": pd.DataFrame(
            {
                "portfolio_name": ["P"],
                "scenario_name": ["Equity Shock"],
                "run_timestamp": [datetime(2024, 2, 6, 12, 0, tzinfo=UTC)],
                "current_portfolio_value": [200.0],
                "shocked_portfolio_value": [185.0],
                "stress_loss": [15.0],
                "stress_return": [0.075],
            }
        )
    }


class _FakeDownloadStreamlit:
    def __init__(self):
        self.download_calls = []

    def download_button(self, label, **kwargs):
        self.download_calls.append({"label": label, **kwargs})


class _FakeRiskPackStreamlit:
    def __init__(self):
        self.subheaders = []
        self.download_calls = []

    def subheader(self, label):
        self.subheaders.append(label)

    def columns(self, count):
        return [_FakeRiskPackColumn(self) for _ in range(count)]


class _FakeRiskPackColumn:
    def __init__(self, parent):
        self.parent = parent

    def download_button(self, label, **kwargs):
        self.parent.download_calls.append({"label": label, **kwargs})
