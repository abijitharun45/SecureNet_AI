"""
SecureNet AI - Sample CSV Generator
====================================
Reads the full CICIoT2023_balanced_test.csv and generates smaller
CSV files suitable for uploading to the deployed app.

Usage:
    python generate_sample_csv.py              # Default: 500 rows
    python generate_sample_csv.py --rows 1000  # Custom size
    python generate_sample_csv.py --rows 200 --output data/tiny_test.csv
"""

import pandas as pd
import argparse
import os

SOURCE_FILE = "data/CICIoT2023_balanced_test.csv"

def generate_sample(rows: int, output: str, stratify: bool = True):
    print(f"📂 Loading source: {SOURCE_FILE}")
    df = pd.read_csv(SOURCE_FILE)
    print(f"   → Original size: {len(df):,} rows, {df.shape[1]} columns")
    print(f"   → File size: {os.path.getsize(SOURCE_FILE) / 1024 / 1024:.1f} MB")

    # Cap rows to dataset size
    rows = min(rows, len(df))

    if stratify and "Label" in df.columns:
        # Stratified sampling: preserves the ratio of attack types
        print(f"🎯 Stratified sampling {rows:,} rows (preserving label distribution)...")
        sample = df.groupby("Label", group_keys=False).apply(
            lambda x: x.sample(n=max(1, int(len(x) / len(df) * rows)), random_state=42)
        )
        # Trim or pad to exact row count
        if len(sample) > rows:
            sample = sample.sample(n=rows, random_state=42)
        elif len(sample) < rows:
            remaining = df.drop(sample.index).sample(n=rows - len(sample), random_state=42)
            sample = pd.concat([sample, remaining])
    else:
        # Random sampling
        print(f"🎲 Random sampling {rows:,} rows...")
        sample = df.sample(n=rows, random_state=42)

    # Shuffle the final result
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    sample.to_csv(output, index=False)
    output_size = os.path.getsize(output) / 1024
    print(f"\n✅ Generated: {output}")
    print(f"   → {len(sample):,} rows, {sample.shape[1]} columns")
    print(f"   → File size: {output_size:.0f} KB")

    # Show label distribution
    if "Label" in sample.columns:
        print(f"\n📊 Label Distribution:")
        dist = sample["Label"].value_counts()
        for label, count in dist.items():
            print(f"   {label}: {count} ({count/len(sample)*100:.1f}%)")

    print(f"\n🚀 Upload '{output}' to your app's Static File Upload mode!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate smaller CSV samples from CICIOT2023 dataset")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows in output (default: 500)")
    parser.add_argument("--output", type=str, default="data/sample_upload.csv", help="Output file path")
    parser.add_argument("--no-stratify", action="store_true", help="Disable stratified sampling")
    args = parser.parse_args()

    if not os.path.exists(SOURCE_FILE):
        print(f"❌ Source file not found: {SOURCE_FILE}")
        print(f"   Place CICIoT2023_balanced_test.csv in the data/ folder first.")
        exit(1)

    generate_sample(args.rows, args.output, stratify=not args.no_stratify)
