# Current mechanical event-blackout calendar

**Status:** verified CPI, Employment Situation and FOMC schedules through
2026-12-31; automatically wired into OANDA pre-trade assessment

## Scope

This component uses schedules only. It does not ingest economic results,
headlines, forecasts, surprises, language or geopolitical information.

The version-one entry blackouts remain:

- CPI and Employment Situation: 30 minutes before through 30 minutes after the
  scheduled 08:30 America/New_York release;
- FOMC statement: 30 minutes before through 60 minutes after the scheduled
  14:00 America/New_York statement.

America/New_York localization occurs before UTC conversion, preserving DST.
Intervals are half-open: the start is blocked and the end is clear.

## Sources and verification

The 2026 CPI and Employment Situation dates were verified on 2026-08-04 from
the [official BLS 2026 release schedule](https://www.bls.gov/schedule/2026/home.htm).
The eight 2026 FOMC meetings were rechecked against the
[Federal Reserve meeting calendar](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm).

Dates live in `src/commodity_ml/features/event_dates.py`; no first-Friday or
other inferred scheduling rule is used. Tests require 12 CPI, 12 Employment
Situation and eight FOMC entries for 2026 and verify an exact August blackout.

## Runtime behavior

`scripts/evaluate_oanda_pretrade_risk.py` now defaults to `--event-state auto`.
It computes the state at the assessment timestamp, records active reason codes,
calendar verification date and coverage end in provenance, and refuses dates
outside verified coverage rather than treating them as calm.

Manual `unknown`, `clear` and `blackout` modes remain available only for
isolated offline diagnostics. Supplying persistent `--paper-state` requires
automatic calendar mode, preventing a paper runtime from bypassing the
verified schedule with a manual clear assertion.

The calendar must be extended and re-verified before 2027 paper operation.
