from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from commodity_ml.intraday.risk_ui import (
    RiskUIError,
    evaluate_manual_request,
    evaluate_read_only_request,
    latest_admitted_calibration,
    normalize_manual_request,
    normalize_trade_request,
)


_SERVER_SPEC = importlib.util.spec_from_file_location(
    "run_risk_ui",
    Path(__file__).resolve().parents[1] / "scripts/run_risk_ui.py",
)
assert _SERVER_SPEC is not None and _SERVER_SPEC.loader is not None
_SERVER_MODULE = importlib.util.module_from_spec(_SERVER_SPEC)
_SERVER_SPEC.loader.exec_module(_SERVER_MODULE)
RiskUIServer = _SERVER_MODULE.RiskUIServer


def _request(**changes):
    trade = {
        "instrument": "XAUUSD",
        "side": "long",
        "order_type": "market",
        "holding_period_minutes": 120,
        "stop_price": 2400,
        "target_price": 2600,
    }
    trade.update(changes)
    return {"trade_intent": trade}


def test_browser_contract_is_small_and_fail_closed():
    result = normalize_trade_request(_request())
    assert result["trade_intent"]["decision_time"] is None
    assert result["trade_intent"]["allow_overnight"] is False

    with pytest.raises(RiskUIError, match="instrument"):
        normalize_trade_request(_request(instrument="EURUSD"))
    with pytest.raises(RiskUIError, match="holding_period"):
        normalize_trade_request(_request(holding_period_minutes=30))
    with pytest.raises(RiskUIError, match="entry_price is required"):
        normalize_trade_request(_request(order_type="limit"))
    with pytest.raises(RiskUIError, match="finite"):
        normalize_trade_request(_request(requested_size=float("inf")))
    with pytest.raises(RiskUIError, match="true or false"):
        normalize_trade_request(_request(allow_overnight="false"))


def _manual_request(**manual_changes):
    manual = {
        "equity": 10_000,
        "available_funds": 9_000,
        "day_start_equity": 10_000,
        "week_start_equity": 10_000,
        "entries_today": 0,
        "kill_switch_active": False,
        "margin_rate": 0.05,
        "commission_per_stake_per_side": 0,
        "no_other_metals_positions_confirmed": True,
    }
    manual.update(manual_changes)
    return {**_request(requested_size=2), "manual_account": manual}


def test_manual_account_contract_is_explicit_and_fail_closed():
    result = normalize_manual_request(_manual_request())
    assert result["trade_intent"]["requested_size"] == 2
    assert result["manual_account"]["margin_rate"] == pytest.approx(0.05)

    with pytest.raises(RiskUIError, match="stake"):
        normalize_manual_request({**_request(), "manual_account": _manual_request()["manual_account"]})
    with pytest.raises(RiskUIError, match="confirm"):
        normalize_manual_request(_manual_request(no_other_metals_positions_confirmed=False))
    with pytest.raises(RiskUIError, match="decimal fraction"):
        normalize_manual_request(_manual_request(margin_rate=5))


def test_oanda_runner_preserves_read_only_boundary(tmp_path, monkeypatch):
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts/evaluate_oanda_pretrade_risk.py").touch()

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps({"order_authority": False, "risk_report": {"decision": "BLOCK"}}),
            encoding="utf-8",
        )
        assert "config/gold.yaml" in command[command.index("--config") + 1]
        assert "--event-state" in command
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("commodity_ml.intraday.risk_ui.subprocess.run", fake_run)
    result = evaluate_read_only_request(_request(), project_root=root)
    assert result["order_authority"] is False
    assert result["risk_report"]["decision"] == "BLOCK"


def test_manual_runner_uses_external_adapter_and_preserves_boundary(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/evaluate_manual_pretrade_risk.py").touch()

    def fake_run(command, **kwargs):
        assert command[1].endswith("evaluate_manual_pretrade_risk.py")
        source = json.loads(Path(command[2]).read_text(encoding="utf-8"))
        assert source["manual_account"]["equity"] == 10_000
        assert "config/gold.yaml" in command[command.index("--config") + 1]
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"order_authority": False, "risk_report": {"decision": "ALLOW"}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("commodity_ml.intraday.risk_ui.subprocess.run", fake_run)
    result = evaluate_manual_request(_manual_request(), project_root=tmp_path)
    assert result["order_authority"] is False


def test_only_a_complete_admitted_calibration_is_auto_selected(tmp_path):
    rejected = tmp_path / "data/processed/current_risk_calibration/20260805T100000Z"
    rejected.mkdir(parents=True)
    (rejected / "release.json").write_text(
        json.dumps({"release_status": "REJECTED"}), encoding="utf-8"
    )
    (rejected / "profile.parquet").write_bytes(b"profile")
    admitted = tmp_path / "data/processed/current_risk_calibration/20260805T110000Z"
    admitted.mkdir(parents=True)
    (admitted / "release.json").write_text(
        json.dumps({"release_status": "ADMITTED"}), encoding="utf-8"
    )
    (admitted / "profile.parquet").write_bytes(b"profile")

    assert latest_admitted_calibration(tmp_path) == (
        (admitted / "profile.parquet").resolve(),
        (admitted / "release.json").resolve(),
    )


def test_oanda_runner_rejects_any_order_authority(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/evaluate_oanda_pretrade_risk.py").touch()

    def fake_run(command, **kwargs):
        Path(command[command.index("--output") + 1]).write_text(
            json.dumps({"order_authority": True, "risk_report": {"decision": "ALLOW"}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("commodity_ml.intraday.risk_ui.subprocess.run", fake_run)
    with pytest.raises(RiskUIError, match="read-only boundary"):
        evaluate_read_only_request(_request(), project_root=tmp_path)


def test_local_server_serves_build_routes_manual_mode_and_rejects_unknown_api(
    tmp_path, monkeypatch
):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Risk UI</title>", encoding="utf-8")
    monkeypatch.setattr(
        _SERVER_MODULE,
        "evaluate_manual_request",
        lambda document, **kwargs: {
            "mode": "MANUAL_GBP_ACCOUNT_OANDA_MARKET_READ_ONLY",
            "order_authority": False,
            "risk_report": {"decision": "ALLOW"},
        },
    )
    try:
        server = RiskUIServer(("127.0.0.1", 0), static, tmp_path)
    except PermissionError:
        pytest.skip("test environment does not permit localhost sockets")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = json.loads(urlopen(f"{base}/api/health", timeout=3).read())
        assert health == {
            "status": "ok",
            "mode": "READ_ONLY_RISK_UI",
            "assessment_modes": [
                "OANDA_V20_READ_ONLY",
                "MANUAL_GBP_ACCOUNT_OANDA_MARKET_READ_ONLY",
            ],
            "order_authority": False,
        }
        assert b"Risk UI" in urlopen(f"{base}/", timeout=3).read()
        manual = Request(
            f"{base}/api/evaluate",
            data=json.dumps({"mode": "manual"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        manual_result = json.loads(urlopen(manual, timeout=3).read())
        assert manual_result["mode"] == "MANUAL_GBP_ACCOUNT_OANDA_MARKET_READ_ONLY"
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}/api/orders", timeout=3)
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
