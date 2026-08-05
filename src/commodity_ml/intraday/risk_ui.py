"""Local-only application layer for the read-only pre-trade risk interface."""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import pandas as pd

from commodity_ml.intraday.pretrade_report import render_pretrade_report_html


SUPPORTED_INSTRUMENTS = {
    "XAUUSD": ("config/gold.yaml", "config/silver.yaml"),
    "XAGUSD": ("config/silver.yaml", "config/gold.yaml"),
}
SUPPORTED_HORIZONS = {60, 120, 180, 240}


class RiskUIError(RuntimeError):
    """A browser request cannot safely be evaluated."""


def latest_admitted_calibration(root: str | Path) -> tuple[Path, Path] | None:
    """Find the newest complete admitted calibration pair, if one exists."""
    calibration_root = Path(root).resolve() / "data/processed/current_risk_calibration"
    admitted: list[tuple[pd.Timestamp, Path, Path]] = []
    if not calibration_root.is_dir():
        return None
    for release_path in calibration_root.glob("*/release.json"):
        profile_path = release_path.with_name("profile.parquet")
        try:
            release_time = pd.to_datetime(
                release_path.parent.name,
                format="%Y%m%dT%H%M%SZ",
                utc=True,
            )
            release = json.loads(release_path.read_text(encoding="utf-8"))
            if release.get("release_status") != "ADMITTED":
                continue
            if release_path.stat().st_size == 0 or profile_path.stat().st_size == 0:
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        admitted.append((release_time, profile_path.resolve(), release_path.resolve()))
    if not admitted:
        return None
    _, profile, release = max(admitted, key=lambda item: item[0])
    return profile, release


def normalize_trade_request(source: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the deliberately small browser input contract."""
    if not isinstance(source, Mapping):
        raise RiskUIError("request body must be a JSON object")
    trade = source.get("trade_intent", source)
    if not isinstance(trade, Mapping):
        raise RiskUIError("trade_intent must be an object")

    instrument = str(trade.get("instrument", "")).upper()
    side = str(trade.get("side", "")).lower()
    order_type = str(trade.get("order_type", "market")).lower()
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise RiskUIError("instrument must be XAUUSD or XAGUSD")
    if side not in {"long", "short"}:
        raise RiskUIError("side must be long or short")
    if order_type not in {"market", "limit", "stop"}:
        raise RiskUIError("order_type must be market, limit or stop")

    horizon = _integer(trade.get("holding_period_minutes"), "holding_period_minutes")
    if horizon not in SUPPORTED_HORIZONS:
        raise RiskUIError("holding_period_minutes must be 60, 120, 180 or 240")

    allow_overnight = trade.get("allow_overnight", False)
    if not isinstance(allow_overnight, bool):
        raise RiskUIError("allow_overnight must be true or false")
    normalized = {
        "intent_id": str(trade.get("intent_id") or f"ui-{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')}"),
        "instrument": instrument,
        "side": side,
        "order_type": order_type,
        "decision_time": _decision_time(trade.get("decision_time")),
        "entry_price": _optional_positive(trade.get("entry_price"), "entry_price"),
        "stop_price": _optional_positive(trade.get("stop_price"), "stop_price"),
        "target_price": _optional_positive(trade.get("target_price"), "target_price"),
        "requested_size": _optional_positive(trade.get("requested_size"), "requested_size"),
        "holding_period_minutes": horizon,
        "allow_overnight": allow_overnight,
        "max_slippage_points_per_side": _optional_nonnegative(
            trade.get("max_slippage_points_per_side"),
            "max_slippage_points_per_side",
        ),
    }
    if order_type != "market" and normalized["entry_price"] is None:
        raise RiskUIError("entry_price is required for limit and stop orders")
    return {"trade_intent": normalized}


def evaluate_read_only_request(
    source: Mapping[str, Any],
    *,
    project_root: str | Path,
    python_executable: str | Path | None = None,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """Run the established OANDA adapter without granting the web server order authority."""
    payload = normalize_trade_request(source)
    root = Path(project_root).resolve()
    instrument = payload["trade_intent"]["instrument"]
    primary_config, portfolio_config = SUPPORTED_INSTRUMENTS[instrument]
    executable = str(python_executable or sys.executable)

    with tempfile.TemporaryDirectory(prefix="commodity-risk-ui-") as temporary:
        temp = Path(temporary)
        input_path = temp / "intent.json"
        output_path = temp / "risk.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        command = [
            executable,
            str(root / "scripts/evaluate_oanda_pretrade_risk.py"),
            str(input_path),
            "--config",
            str(root / primary_config),
            "--portfolio-config",
            str(root / portfolio_config),
            "--event-state",
            "auto",
            "--output",
            str(output_path),
        ]
        paper_state = root / "data/runtime/oanda_paper_state.json"
        if paper_state.is_file():
            command.extend(("--paper-state", str(paper_state)))
        calibration = latest_admitted_calibration(root)
        if calibration is not None:
            profile, release = calibration
            command.extend(
                (
                    "--calibration-profile",
                    str(profile),
                    "--calibration-release",
                    str(release),
                )
            )
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RiskUIError("OANDA risk assessment timed out") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            message = detail[-1] if detail else "read-only OANDA assessment failed"
            raise RiskUIError(message.removeprefix("Read-only OANDA assessment refused: "))
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RiskUIError("risk engine did not produce a valid report") from exc
    if result.get("order_authority") is not False:
        raise RiskUIError("risk engine violated the read-only boundary")
    return result


def render_request_report(document: Mapping[str, Any]) -> str:
    return render_pretrade_report_html(document)


def _decision_time(value: Any) -> str:
    timestamp = pd.Timestamp(value) if value else pd.Timestamp.now(tz="UTC")
    if timestamp.tzinfo is None:
        raise RiskUIError("decision_time must include a timezone")
    return timestamp.tz_convert("UTC").isoformat()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise RiskUIError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RiskUIError(f"{name} must be an integer") from exc
    return number


def _optional_positive(value: Any, name: str) -> float | None:
    number = _optional_number(value, name)
    if number is not None and number <= 0:
        raise RiskUIError(f"{name} must be positive")
    return number


def _optional_nonnegative(value: Any, name: str) -> float | None:
    number = _optional_number(value, name)
    if number is not None and number < 0:
        raise RiskUIError(f"{name} cannot be negative")
    return number


def _optional_number(value: Any, name: str) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise RiskUIError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RiskUIError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise RiskUIError(f"{name} must be finite")
    return number
