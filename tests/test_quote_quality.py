import pandas as pd

from commodity_ml.intraday.quote_quality import summarize_quote_collection


def test_quote_collection_summary_preserves_duplicates_and_flags_invalid_events():
    quotes = pd.DataFrame(
        {
            "instrument": ["XAU_USD", "XAU_USD", "XAG_USD"],
            "broker_time": [
                "2026-08-04T10:00:00Z",
                "2026-08-04T10:00:00Z",
                "2026-08-04T10:00:01Z",
            ],
            "received_at": [
                "2026-08-04T10:00:00.020Z",
                "2026-08-04T10:00:00.030Z",
                "2026-08-04T10:00:01.040Z",
            ],
            "bid": [2400.0, 2400.1, 30.0],
            "ask": [2400.3, 2400.4, 30.03],
        }
    )
    events = pd.DataFrame(
        {"event_type": ["CONNECTED", "HEARTBEAT", "SERVER_ERROR", "DISCONNECTED_CLEAN"]}
    )

    report = summarize_quote_collection(quotes, events, schema_version=2)

    assert report.total_quotes == 3
    assert report.path_invalidating_events == 2
    assert report.unexpected_invalidating_events == 1
    assert report.recovered_invalidating_events == 0
    assert report.unresolved_invalidating_events == 1
    assert [item.instrument for item in report.instruments] == ["XAG_USD", "XAU_USD"]
    gold = report.instruments[1]
    assert gold.duplicate_broker_times == 1
    assert gold.crossed_quotes == 0
    assert gold.median_receive_latency_ms == 25.0


def test_successful_reconnect_marks_prior_stream_fault_as_recovered():
    quotes = pd.DataFrame(
        {
            "instrument": ["XAU_USD"],
            "broker_time": ["2026-08-04T10:00:00Z"],
            "received_at": ["2026-08-04T10:00:00.020Z"],
            "bid": [2400.0],
            "ask": [2400.2],
        }
    )
    events = pd.DataFrame(
        {"event_type": ["SERVER_ERROR", "DISCONNECTED_ERROR", "CONNECTED"]}
    )

    report = summarize_quote_collection(quotes, events, schema_version=2)

    assert report.unexpected_invalidating_events == 2
    assert report.recovered_invalidating_events == 2
    assert report.unresolved_invalidating_events == 0
