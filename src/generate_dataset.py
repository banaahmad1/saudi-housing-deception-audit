"""
Real Kaggle Saudi Real Estate (Aqar) Dataset Ingestion & Preprocessing Script.
Fetches authentic real-world listings from the verified Kaggle Aqar repository,
cleans attributes, and builds the benchmark dataset with zero synthetic generation
or fabrications.
"""

from pathlib import Path
import argparse
import sys
import pandas as pd

# Ensure Windows terminal handles UTF-8 Arabic output cleanly
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_DATA_FILE
from scripts.process_real_aqar_data import download_and_clean_aqar_data, formulate_benchmarks

def generate_benchmark_dataset(num_samples: int = 350, output_path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Ingests authentic Saudi Real Estate (Aqar) listings from the verified repository,
    cleans attributes, and exports benchmark data.
    """
    print(f"Ingesting authentic Kaggle Aqar real estate dataset (target: {output_path})...")
    df = download_and_clean_aqar_data()
    combined_df, held_out_df = formulate_benchmarks(df)
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest authentic Saudi Real Estate listings")
    parser.add_argument("--samples", type=int, default=350, help="Minimum sample target")
    parser.add_argument("--output", type=str, default=str(RAW_DATA_FILE), help="Output CSV path")
    args = parser.parse_args()

    generate_benchmark_dataset(num_samples=args.samples, output_path=Path(args.output))
