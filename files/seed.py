#!/usr/bin/env python3
"""
seed.py – Initialise and populate the real estate SQLite database.

Usage
-----
    python seed.py               # init schema + insert 120 listings (skip if data exists)
    python seed.py --count 300   # insert 300 listings
    python seed.py --force       # wipe existing data and re-seed
    python seed.py --stats       # print DB stats and exit (no seeding)

This script is safe to run repeatedly; without --force it is a no-op when
the database already contains data.
"""

import argparse
import sys
from database import DB_PATH, get_stats, init_db, seed_mock_data


def _print_stats() -> None:
    stats = get_stats()
    hood_rates = stats.get("neighborhood_avg_ppsqm", {})

    print(f"\n{'─'*52}")
    print(f"  Database  : {DB_PATH}")
    print(f"  Refined   : {stats['total_refined']:>6,} rows")
    print(f"  Raw       : {stats['total_raw']:>6,} rows")
    print(f"  Low-qual  : {stats['low_quality_count']:>6,} rows  ({stats['outlier_pct']:.1f}%)")
    print(f"  Updated   : {str(stats['last_update'] or '—')[:19]}")
    print(f"{'─'*52}")

    if hood_rates:
        print(f"\n  Avg price/sqm by neighborhood:")
        for hood, avg in sorted(hood_rates.items(), key=lambda x: -x[1]):
            bar = "█" * int(avg / 2_000)
            print(f"    {hood:<20s} {avg:>10,.0f} EGP/sqm  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the Real Estate Analytics Hub SQLite database.",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=120,
        metavar="N",
        help="Number of mock listings to insert (default: 120)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Delete all existing rows and re-seed from scratch",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Print database statistics and exit without seeding",
    )
    args = parser.parse_args()

    # Always ensure schema exists
    init_db()
    print(f"✅ Schema ready  →  {DB_PATH}")

    if args.stats:
        _print_stats()
        sys.exit(0)

    inserted = seed_mock_data(n=args.count, force=args.force)

    if inserted == 0:
        print(
            "ℹ️  Database already contains data — skipping seed.\n"
            "   Use --force to wipe and re-seed."
        )
    else:
        print(f"🌱 Inserted {inserted} listings into raw_listings + refined_listings.")

    _print_stats()


if __name__ == "__main__":
    main()
