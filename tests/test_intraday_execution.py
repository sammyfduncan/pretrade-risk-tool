"""Synthetic tests for the frozen intraday trade contract."""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.intraday.execution import (
    CandidateRejected,
    EntryBlackout,
    IntradayExecutionConfig,
    SessionCutoffOverride,
    TradeCandidate,
    labels_from_ledger,
    resolve_trade,
    simulate_candidates,
)


def make_candles(
    rows: list[tuple[float, float, float, float]],
    start: str = "2026-01-05 07:00",
) -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(rows), freq="30min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def add_quotes(
    candles: pd.DataFrame,
    *,
    bid_offset: float = -0.05,
    ask_offset: float = 0.05,
) -> pd.DataFrame:
    quoted = candles.copy()
    for field in ("open", "high", "low", "close"):
        quoted[f"bid_{field}"] = quoted[field] + bid_offset
        quoted[f"ask_{field}"] = quoted[field] + ask_offset
    return quoted


def candidate(
    signal_time: pd.Timestamp,
    *,
    side: str = "long",
    stop: float = 0.01,
    target: float = 0.01,
    bars: int = 4,
    strategy: str = "test",
) -> TradeCandidate:
    return TradeCandidate(
        signal_time=signal_time,
        side=side,
        stop_distance_pct=stop,
        target_distance_pct=target,
        max_holding_bars=bars,
        strategy=strategy,
    )


def test_entry_and_barriers_anchor_to_next_bar_open():
    candles = make_candles(
        [
            (100.0, 100.5, 99.5, 100.0),  # signal closes at 100
            (105.0, 106.2, 104.5, 105.2),  # executable entry opens at 105
        ]
    )

    outcome = resolve_trade(candles, candidate(candles.index[0], bars=1))

    assert outcome.entry_time == candles.index[1]
    assert outcome.entry_price == 105.0
    assert outcome.target_price == pytest.approx(106.05)
    assert outcome.exit_price == pytest.approx(106.05)
    assert outcome.exit_reason == "target"


def test_absolute_atr_distance_is_preserved_across_entry_gap():
    candles = make_candles(
        [
            (100.0, 100.5, 99.5, 100.0),
            (105.0, 108.2, 103.5, 107.0),
        ]
    )
    atr_candidate = TradeCandidate(
        signal_time=candles.index[0],
        side="long",
        stop_distance_pct=None,
        target_distance_pct=None,
        max_holding_bars=1,
        strategy="atr_rule",
        stop_distance_abs=2.0,
        target_r_multiple=1.5,
    )

    outcome = resolve_trade(candles, atr_candidate)

    assert outcome.entry_price == 105.0
    assert outcome.stop_price == 103.0
    assert outcome.target_price == 108.0
    assert outcome.initial_r_pct == pytest.approx(2 / 105)
    assert outcome.exit_reason == "target"


def test_actual_high_low_trigger_a_stop_even_when_close_does_not():
    candles = make_candles(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.3, 98.8, 100.1),
        ]
    )

    outcome = resolve_trade(candles, candidate(candles.index[0], bars=1))

    assert outcome.exit_reason == "stop"
    assert outcome.exit_price == pytest.approx(99.0)


@pytest.mark.parametrize("side", ["long", "short"])
def test_same_bar_stop_and_target_resolve_adversely(side):
    candles = make_candles(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 102.0, 98.0, 100.0),
        ]
    )

    outcome = resolve_trade(candles, candidate(candles.index[0], side=side, bars=1))

    assert outcome.exit_reason == "stop"
    expected = 99.0 if side == "long" else 101.0
    assert outcome.exit_price == pytest.approx(expected)
    assert outcome.label == 0


def test_short_target_is_symmetric_and_net_profitable():
    candles = make_candles(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 98.5, 99.2),
        ]
    )

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], side="short", stop=0.01, target=0.01, bars=1),
    )

    assert outcome.exit_reason == "target"
    assert outcome.exit_price == pytest.approx(99.0)
    assert outcome.gross_return == pytest.approx(0.01)
    assert outcome.net_return == pytest.approx(0.0098)
    assert outcome.label == 1


def test_round_trip_cost_determines_the_label():
    candles = make_candles(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.01),
        ]
    )
    cfg = IntradayExecutionConfig(cost_pct_per_side=0.0001)

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], stop=0.10, target=0.10, bars=1),
        cfg,
    )

    assert outcome.exit_reason == "timeout"
    assert outcome.gross_return == pytest.approx(0.0001)
    assert outcome.net_return == pytest.approx(-0.0001)
    assert outcome.label == 0


@pytest.mark.parametrize(
    ("side", "expected_entry", "expected_exit"),
    [
        ("long", 100.05, 99.95),
        ("short", 99.95, 100.05),
    ],
)
def test_bid_ask_quotes_embed_spread_in_both_directions(
    side,
    expected_entry,
    expected_exit,
):
    candles = add_quotes(
        make_candles(
            [
                (100.0, 100.1, 99.9, 100.0),
                (100.0, 100.1, 99.9, 100.0),
            ]
        )
    )
    cfg = IntradayExecutionConfig(cost_pct_per_side=0.0)

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], side=side, stop=0.10, target=0.10, bars=1),
        cfg,
    )

    assert outcome.price_source == "quotes"
    assert outcome.entry_price == pytest.approx(expected_entry)
    assert outcome.exit_price == pytest.approx(expected_exit)
    assert outcome.gross_return < 0
    assert outcome.net_return == outcome.gross_return
    assert outcome.label == 0


def test_long_barriers_use_bid_path_not_mid_path():
    candles = make_candles(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.2, 99.3, 100.0),
        ]
    )
    candles = add_quotes(candles, bid_offset=-0.25, ask_offset=0.25)
    cfg = IntradayExecutionConfig(cost_pct_per_side=0.0)

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], stop=0.01, target=0.10, bars=1),
        cfg,
    )

    # Entry is ask 100.25, so the stop is 99.2475. Mid low 99.30 does not
    # touch it, while executable bid low 99.05 does.
    assert outcome.entry_price == pytest.approx(100.25)
    assert outcome.stop_price == pytest.approx(99.2475)
    assert outcome.exit_price == pytest.approx(99.2475)
    assert outcome.exit_reason == "stop"


def test_explicit_cost_is_additional_to_observed_spread():
    candles = add_quotes(
        make_candles(
            [
                (100.0, 100.1, 99.9, 100.0),
                (100.0, 100.1, 99.9, 100.0),
            ]
        )
    )
    cfg = IntradayExecutionConfig(cost_pct_per_side=0.0002)

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], stop=0.10, target=0.10, bars=1),
        cfg,
    )

    assert outcome.net_return == pytest.approx(outcome.gross_return - 0.0004)


def test_partial_bid_ask_columns_are_rejected():
    candles = make_candles(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 100.1, 99.9, 100.0),
        ]
    )
    candles["bid_open"] = candles["open"] - 0.05

    with pytest.raises(ValueError, match="every quote OHLC"):
        resolve_trade(candles, candidate(candles.index[0], bars=1))


def test_crossed_bid_ask_quotes_are_rejected():
    candles = add_quotes(
        make_candles(
            [
                (100.0, 100.1, 99.9, 100.0),
                (100.0, 100.1, 99.9, 100.0),
            ]
        )
    )
    candles.loc[candles.index[1], ["ask_open", "ask_low"]] = [99.90, 99.80]

    with pytest.raises(ValueError, match="crossed quote"):
        resolve_trade(candles, candidate(candles.index[0], bars=1))


def test_daily_cutoff_exits_at_last_complete_bar():
    rows = [(100.0, 100.1, 99.9, 100.0)] * 6
    candles = make_candles(rows, start="2026-01-05 18:30")

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], stop=0.20, target=0.20, bars=10),
    )

    assert outcome.entry_time == pd.Timestamp("2026-01-05 19:00", tz="UTC")
    assert outcome.exit_time == pd.Timestamp("2026-01-05 21:30", tz="UTC")
    assert outcome.exit_reason == "cutoff"
    assert outcome.bars_held == 5


def test_london_cutoff_follows_daylight_saving_time():
    rows = [(100.0, 100.1, 99.9, 100.0)] * 7
    candles = make_candles(rows, start="2026-06-08 17:30")

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], stop=0.20, target=0.20, bars=10),
    )

    # In June, 21:30 Europe/London is 20:30 UTC.
    assert outcome.entry_time == pd.Timestamp("2026-06-08 18:00", tz="UTC")
    assert outcome.exit_time == pd.Timestamp("2026-06-08 20:30", tz="UTC")
    assert outcome.exit_reason == "cutoff"


def test_entry_at_or_after_1930_london_is_rejected():
    candles = make_candles(
        [(100.0, 100.1, 99.9, 100.0)] * 3,
        start="2026-01-05 18:30",
    )

    with pytest.raises(CandidateRejected, match="outside_entry_window"):
        resolve_trade(candles, candidate(candles.index[1]))


def test_scheduled_blackout_rejects_entry_without_queueing():
    candles = make_candles([(100.0, 100.1, 99.9, 100.0)] * 4)
    cfg = IntradayExecutionConfig(
        entry_blackouts=(
            EntryBlackout(
                start=pd.Timestamp("2026-01-05 07:15", tz="UTC"),
                end=pd.Timestamp("2026-01-05 08:00", tz="UTC"),
                reason="cpi",
            ),
        )
    )

    with pytest.raises(CandidateRejected, match="entry_blackout:cpi"):
        resolve_trade(candles, candidate(candles.index[0]), cfg)


def test_early_close_override_forces_flat_at_adjusted_cutoff():
    rows = [(100.0, 100.1, 99.9, 100.0)] * 9
    candles = make_candles(rows, start="2026-01-05 14:30")
    cfg = IntradayExecutionConfig(
        cutoff_overrides=(
            SessionCutoffOverride(
                session_date=pd.Timestamp("2026-01-05").date(),
                flat_cutoff=pd.Timestamp("17:30").time(),
            ),
        )
    )

    outcome = resolve_trade(
        candles,
        candidate(candles.index[0], stop=0.20, target=0.20, bars=10),
        cfg,
    )

    assert outcome.entry_time == pd.Timestamp("2026-01-05 15:00", tz="UTC")
    assert outcome.exit_time == pd.Timestamp("2026-01-05 17:30", tz="UTC")
    assert outcome.exit_reason == "cutoff"


def test_candidate_is_not_carried_across_a_missing_bar():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-01-05 07:00", tz="UTC"),
            pd.Timestamp("2026-01-05 08:00", tz="UTC"),
        ]
    )
    candles = pd.DataFrame(
        [(100.0, 100.1, 99.9, 100.0)] * 2,
        columns=["open", "high", "low", "close"],
        index=index,
    )

    with pytest.raises(CandidateRejected, match="non_contiguous_entry"):
        resolve_trade(candles, candidate(candles.index[0]))


def test_simulator_rejects_overlapping_candidates_without_queueing():
    candles = make_candles([(100.0, 100.1, 99.9, 100.0)] * 9)
    candidates = [
        candidate(candles.index[0], stop=0.20, target=0.20, bars=4, strategy="first"),
        candidate(candles.index[1], stop=0.20, target=0.20, bars=1, strategy="blocked"),
        candidate(candles.index[4], stop=0.20, target=0.20, bars=1, strategy="next"),
    ]

    ledger = simulate_candidates(candles, candidates)

    assert ledger["status"].tolist() == ["executed", "rejected", "executed"]
    assert ledger.loc[1, "reject_reason"] == "position_open"
    assert ledger.loc[2, "entry_time"] == pd.Timestamp("2026-01-05 09:30", tz="UTC")


def test_simulator_enforces_executed_daily_and_strategy_entry_caps():
    candles = make_candles([(100.0, 100.1, 99.9, 100.0)] * 12)
    p1_candidates = [
        candidate(candles.index[0], stop=0.20, target=0.20, bars=1, strategy="P1"),
        candidate(candles.index[2], stop=0.20, target=0.20, bars=1, strategy="P1"),
    ]
    strategy_ledger = simulate_candidates(candles, p1_candidates)
    assert strategy_ledger["status"].tolist() == ["executed", "rejected"]
    assert strategy_ledger.loc[1, "reject_reason"] == "strategy_daily_entry_limit"

    uncapped_strategies = [
        candidate(
            candles.index[pos],
            stop=0.20,
            target=0.20,
            bars=1,
            strategy=f"custom_{pos}",
        )
        for pos in (0, 2, 4, 6)
    ]
    daily_ledger = simulate_candidates(candles, uncapped_strategies)
    assert daily_ledger["status"].tolist() == [
        "executed",
        "executed",
        "executed",
        "rejected",
    ]
    assert daily_ledger.loc[3, "reject_reason"] == "daily_entry_limit"


def test_labels_are_derived_from_the_exact_executed_net_returns():
    candles = make_candles(
        [
            (100.0, 100.1, 99.9, 100.0),
            (100.0, 101.2, 99.8, 101.0),  # first candidate target
            (101.0, 101.1, 100.9, 101.0),
            (101.0, 101.1, 99.8, 100.0),  # second candidate stop
        ]
    )
    candidates = [
        candidate(candles.index[0], bars=1, strategy="winner"),
        candidate(candles.index[2], bars=1, strategy="loser"),
    ]

    ledger = simulate_candidates(candles, candidates)
    labels = labels_from_ledger(ledger)

    assert labels.tolist() == [1, 0]
    executed = ledger.loc[ledger["status"] == "executed"]
    np.testing.assert_array_equal(
        labels.to_numpy(),
        (executed["net_return"].to_numpy() > 0).astype(np.int8),
    )
