#!/usr/bin/env python3
"""
split_csv_into_batches.py

Usage:
    ./split_csv_into_batches.py <CSV_FILE> <OUT_DIR> <NUM_BATCHES>

This will read CSV_FILE (including its header) and write NUM_BATCHES files
into OUT_DIR named 01.csv, 02.csv, … each containing an equal chunk of rows
(with the header repeated).
"""
import sys
import os
import math
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    csv_file = sys.argv[1]
    out_dir   = sys.argv[2]
    num_batches = int(sys.argv[3])

    # --- load & split ---
    df = pd.read_csv(csv_file, header=None)                                # read entire CSV (header becomes df.columns)
    chunks = np.array_split(df, num_batches)                               # roughly equal splits

    # --- ensure output directory exists ---
    os.makedirs(out_dir, exist_ok=True)

    # --- write each batch ---
    for idx, chunk in enumerate(chunks, start=1):
        batch_name = f"{idx:02d}.csv"
        out_path = os.path.join(out_dir, batch_name)
        chunk.to_csv(out_path, index=False, header=False)                  # includes header, no Pandas index
        print(f"Wrote {len(chunk)} rows ➔ {out_path}")

if __name__ == "__main__":
    main()
