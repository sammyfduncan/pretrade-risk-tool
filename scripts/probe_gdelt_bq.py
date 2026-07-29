"""Inspect the GDELT BigQuery public dataset before committing to a query design.

Run this FIRST, once `gcloud auth application-default login` has been done. It
answers the three questions that decide how features/gdelt.py must be rewritten,
and it is deliberately read-only and (almost) free:

  1. SCHEMA  — which table, which columns, how is it partitioned?
  2. COST    — a dry run reports bytes scanned WITHOUT running the query.
               BigQuery bills on bytes scanned in the columns you touch, and
               the free tier is 1TB/month, so an 8-year GKG scan is worth
               measuring before it is spent.
  3. CONTENT — what do DocumentIdentifier / Extras / V2Themes actually hold?
               DOC 2.0 searched article full text; GKG has no full text, so the
               matching predicate has to be built from whatever IS here. This
               is the methodological decision, and it should be made by reading
               real rows rather than from memory of the schema.

Nothing here writes a snapshot or reads an evaluation number.

Usage:
    python scripts/probe_gdelt_bq.py --project ml-commodity
"""

import argparse

CANDIDATE_TABLES = (
    # Partitioned first: if it exists it is far cheaper to query by date, which
    # matters because the alternative scans the whole archive for every pull.
    "gdelt-bq.gdeltv2.gkg_partitioned",
    "gdelt-bq.gdeltv2.gkg",
)

# A short, cheap window. The point is to see the SHAPE of real rows, not to
# gather data — a single day is enough to decide how matching must work.
SAMPLE_START = "2021-02-01"
SAMPLE_END = "2021-02-02"


def describe_schema(client, table_id: str) -> bool:
    """Print one table's columns. Returns False if it does not exist."""
    from google.cloud.exceptions import NotFound

    try:
        table = client.get_table(table_id)
    except NotFound:
        print(f"  {table_id}: NOT FOUND")
        return False

    partitioning = table.time_partitioning
    print(f"\n=== {table_id} ===")
    print(f"  rows: {table.num_rows:,}   size: {table.num_bytes / 1e12:.2f} TB")
    print(f"  partitioned on: {partitioning.field or '_PARTITIONTIME' if partitioning else 'NOT PARTITIONED'}")
    print("  columns:")
    for field in table.schema:
        print(f"    {field.name:<28} {field.field_type}")
    return True


def dry_run_cost(client, sql: str, label: str) -> None:
    """Report bytes scanned without executing. This is the number that decides
    whether a query design fits in the free tier."""
    from google.cloud import bigquery

    job = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True,
                                                              use_query_cache=False))
    gb = job.total_bytes_processed / 1e9
    print(f"  {label}: {gb:,.1f} GB scanned "
          f"({gb / 1000:.1%} of the 1TB/month free tier)")


def sample_rows(client, table_id: str) -> None:
    """Read a handful of real rows so the matching predicate is designed
    against what the fields actually contain."""
    sql = f"""
    SELECT DocumentIdentifier, V2Tone, SUBSTR(V2Themes, 1, 200) AS themes_head,
           SUBSTR(Extras, 1, 300) AS extras_head
    FROM `{table_id}`
    WHERE DATE(_PARTITIONTIME) BETWEEN '{SAMPLE_START}' AND '{SAMPLE_END}'
      AND LOWER(DocumentIdentifier) LIKE '%silver%'
    LIMIT 10
    """
    print(f"\n=== sample rows mentioning 'silver' ({SAMPLE_START}) ===")
    for row in client.query(sql).result():
        print(f"\n  URL:    {row.DocumentIdentifier[:120]}")
        print(f"  V2Tone: {row.V2Tone}")
        print(f"  themes: {row.themes_head}")
        print(f"  extras: {(row.extras_head or '')[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True,
                        help="your Google Cloud project id (billing/quota owner; "
                             "the DATA lives in the public gdelt-bq project)")
    parser.add_argument("--skip-sample", action="store_true",
                        help="schema + cost only; do not run the sampling query")
    args = parser.parse_args()

    from google.cloud import bigquery

    client = bigquery.Client(project=args.project)
    print(f"Authenticated. Billing project: {client.project}\n")
    print("=== candidate tables ===")

    table_id = None
    for candidate in CANDIDATE_TABLES:
        if describe_schema(client, candidate) and table_id is None:
            table_id = candidate
    if table_id is None:
        raise SystemExit("No GDELT GKG table reachable — check the project and auth.")

    print(f"\n=== cost estimates against {table_id} (dry run, nothing executed) ===")
    one_day = f"""
    SELECT DATE(_PARTITIONTIME) AS day, COUNT(*) AS n, AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) AS tone
    FROM `{table_id}`
    WHERE DATE(_PARTITIONTIME) BETWEEN '{SAMPLE_START}' AND '{SAMPLE_END}'
    GROUP BY day
    """
    full_range = one_day.replace(f"'{SAMPLE_START}' AND '{SAMPLE_END}'",
                                 "'2018-01-01' AND '2026-07-01'")
    dry_run_cost(client, one_day, "1 day, tone+count")
    dry_run_cost(client, full_range, "8.5 years, tone+count")

    if not args.skip_sample:
        sample_rows(client, table_id)

    print("\nNext: decide the matching predicate from the rows above "
          "(URL slug vs Extras page title vs V2Themes), then rewrite "
          "features/gdelt.py's fetch layer. Everything downstream of the "
          "snapshot — derive_daily, alignment, the 3 features — is unchanged.")


if __name__ == "__main__":
    main()
