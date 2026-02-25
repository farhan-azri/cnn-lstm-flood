import argparse
import subprocess
import sys

from extract_hourly_rainfall import extract_hourly_rainfall
from extract_daily_river_discharge import extract_daily_river_discharge
from feature_engineering import build_features


def main():
    parser = argparse.ArgumentParser(description="Hourly rainfall + daily discharge pipeline")
    parser.add_argument("--step", choices=["all", "extract", "features", "train"], default="all")
    parser.add_argument("--start_date", default="2020-01-01")
    parser.add_argument("--end_date", default="2025-10-31")
    args = parser.parse_args()

    if args.step in ("extract", "all"):
        print("📥 Extracting hourly rainfall/weather...")
        extract_hourly_rainfall(start_date=args.start_date, end_date=args.end_date)
        print("📥 Extracting daily river discharge...")
        extract_daily_river_discharge(start_date=args.start_date, end_date=args.end_date)

    if args.step in ("features", "all"):
        print("⚙️ Building daily features (hourly→daily + merge)...")
        build_features()

    if args.step in ("train", "all"):
        print("🤖 Training CNN-LSTM model...")
        subprocess.run([sys.executable, "model_cnn_lstm.py"], check=True)

    print("✅ Done.")


if __name__ == "__main__":
    main()

# python main.py --step all --start_date 2020-01-01 --end_date 2025-10-31