"""
Convert eod_vol_snap / hertzsoy.XX.VOLS wide format to flat CSV.

Outputs to stdout (pipe or redirect) and copies to clipboard for
pasting into the Streamlit dashboard.

Usage:
    python scripts/snap_to_csv.py                           # uses data/eod_vol_snap.csv
    python scripts/snap_to_csv.py hertzsoy.22.VOLS.csv      # specify file
    python scripts/snap_to_csv.py hertzsoy.22.VOLS.csv -o out.csv  # save to file
"""

import sys
import os
import argparse

# Add project root to path so we can import the parser
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))

from update_volskew_clean import parse_csv_file
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Convert vol snap to flat CSV")
    parser.add_argument("input", nargs="?", default=None,
                        help="Input file (default: data/eod_vol_snap.csv)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file (default: print to stdout + clipboard)")
    parser.add_argument("--latest", action="store_true",
                        help="Only output the most recent date")
    args = parser.parse_args()

    # Resolve input path
    if args.input is None:
        input_file = os.path.join(PROJECT_DIR, 'data', 'eod_vol_snap.csv')
    elif os.path.isabs(args.input):
        input_file = args.input
    else:
        input_file = os.path.join(PROJECT_DIR, args.input)

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Parse
    records = parse_csv_file(input_file)
    if not records:
        print("No records parsed.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)

    if args.latest:
        latest = df['date'].max()
        df = df[df['date'] == latest]

    df = df.sort_values(['date', 'commodity', 'expiry'], ascending=[False, True, True])

    # Output
    csv_text = df.to_csv(index=False)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows to {args.output}", file=sys.stderr)
    else:
        print(csv_text, end='')

    # Try to copy to clipboard
    try:
        import subprocess
        proc = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
        proc.communicate(csv_text.encode('utf-8'))
        print(f"\n({len(df)} rows copied to clipboard)", file=sys.stderr)
    except Exception:
        pass


if __name__ == '__main__':
    main()
