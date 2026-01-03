import pandas as pd
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# --- CONFIGURATION ---
DEFAULT_INPUT = 'CICIoT2023_balanced_test.csv'
DEFAULT_OUTPUT = 'test_mini.csv'
DEFAULT_ROWS = 5000

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_subset(input_path: str, output_path: str, sample_size: int) -> bool:
    """
    Reads the top N rows from a large CSV dataset and saves them to a smaller file
    for testing and demonstration purposes.

    Args:
        input_path (str): Path to the source large CSV file.
        output_path (str): Path where the subset CSV will be saved.
        sample_size (int): Number of rows to read.

    Returns:
        bool: True if successful, False otherwise.
    """
    source_file = Path(input_path)

    if not source_file.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.warning("Please ensure the large dataset is in the project root directory.")
        return False

    try:
        logger.info(f"Reading first {sample_size} rows from '{input_path}'...")

        # Read only necessary rows to optimize memory usage
        df = pd.read_csv(input_path, nrows=sample_size)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Save to disk
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Success! Generated mini-dataset: '{output_path}' ({len(df)} rows)")
        return True

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return False


def main():
    """
    Main execution entry point. Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a mini-dataset from a large CSV file.")

    parser.add_argument(
        '--input',
        type=str,
        default=DEFAULT_INPUT,
        help=f"Path to input CSV (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to output CSV (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        '--rows',
        type=int,
        default=DEFAULT_ROWS,
        help=f"Number of rows to extract (default: {DEFAULT_ROWS})"
    )

    args = parser.parse_args()

    success = create_subset(args.input, args.output, args.rows)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()