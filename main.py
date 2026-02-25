import argparse
import subprocess
import sys
from pathlib import Path

from extract_hourly_rainfall import extract_hourly_rainfall
from extract_daily_river_discharge import extract_daily_river_discharge
from feature_engineering import build_features


def main():
    parser = argparse.ArgumentParser(description="Hourly rainfall + daily discharge pipeline")
    parser.add_argument("--step", choices=["all", "extract", "features", "train"], default="all")
    parser.add_argument("--start_date", default="2020-01-01")
    parser.add_argument("--end_date", default="2025-10-31")

    parser.add_argument("--weather_hourly_path", default="data/weather_hourly.csv")
    parser.add_argument("--flood_daily_path", default="data/flood_daily.csv")
    parser.add_argument("--features_out_path", default="data/features_daily.csv")

    args = parser.parse_args()

    if args.step in ("extract", "all"):
        print("📥 Extracting hourly rainfall/weather...")
        extract_hourly_rainfall(
            start_date=args.start_date,
            end_date=args.end_date,
            out_path=args.weather_hourly_path
        )

        print("📥 Extracting daily river discharge...")
        extract_daily_river_discharge(
            start_date=args.start_date,
            end_date=args.end_date,
            out_path=args.flood_daily_path
        )

    if args.step in ("features", "all"):
        print("⚙️ Building daily features (hourly → daily + merge)...")
        build_features(
            weather_hourly_path=args.weather_hourly_path,
            flood_daily_path=args.flood_daily_path,
            out_path=args.features_out_path,
            verbose=True
        )

        if not Path(args.features_out_path).exists():
            raise FileNotFoundError(f"❌ Feature file not created: {args.features_out_path}")

    if args.step in ("train", "all"):
        print("🤖 Training CNN-LSTM model...")
        subprocess.run([sys.executable, "model_cnn_lstm.py"], check=True)

    print("✅ Done. For EDA + prediction dashboard, run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()