import argparse
import subprocess
import sys
from pathlib import Path

from extract_hourly_rainfall import extract_hourly_rainfall
from extract_daily_river_discharge import extract_daily_river_discharge
from combine_daily_river_weather_hourly import combine_daily_river_weather_hourly
from extract_weather_forecast import extract_seasonal_hourly
from extract_weather_ensemble_forecast import extract_ensemble_hourly
from feature_engineering import build_features


def main():
    parser = argparse.ArgumentParser(description="Hourly rainfall + daily discharge pipeline")

    # ✅ Added "predict"
    parser.add_argument(
        "--step",
        choices=["all", "extract", "combine", "features", "train", "predict"],
        default="all"
    )

    parser.add_argument("--start_date", default="2010-01-01")
    parser.add_argument("--end_date", default="2026-03-01")

    parser.add_argument("--weather_hourly_path", default="data/weather_hourly.csv")
    parser.add_argument("--weather_forecast_path", default="data/weather_forecast_hourly.csv")
    parser.add_argument("--weather_ensemble_path", default="data/weather_ensemble_forecast_hourly.csv")
    parser.add_argument("--flood_daily_path", default="data/flood_daily.csv")
    parser.add_argument("--combine_out_path", default="data/combine_daily_river_weather_hourly.csv")
    parser.add_argument("--features_out_path", default="data/features_daily.csv")

    args = parser.parse_args()

    # ============================================================
    # EXTRACT
    # ============================================================
    if args.step in ("extract", "all"):
        print("📥 Extracting hourly rainfall/weather...")
        extract_hourly_rainfall(
            start_date=args.start_date,
            end_date=args.end_date,
            out_path=args.weather_hourly_path
        )

        print("📥 Extracting seasonal weather forecast...")
        extract_seasonal_hourly(
            out_path=args.weather_forecast_path
        )

        print("📥 Extracting ensemble weather forecast...")
        extract_ensemble_hourly(
            out_path=args.weather_ensemble_path
        )

        print("📥 Extracting daily river discharge...")
        extract_daily_river_discharge(
            start_date=args.start_date,
            end_date=args.end_date,
            out_path=args.flood_daily_path
        )

    # ============================================================
    # COMBINE
    # ============================================================
    if args.step in ("combine", "all"):
        print("🔗 Combining rainfall + discharge into single dataset...")

        combine_daily_river_weather_hourly(
            flood_daily_path=args.flood_daily_path,
            weather_hourly_path=args.weather_hourly_path,
            out_path=args.combine_out_path,
            verbose=True
        )

        if not Path(args.combine_out_path).exists():
            raise FileNotFoundError(
                f"❌ Combined file not created: {args.combine_out_path}"
            )

    # ============================================================
    # FEATURES
    # ============================================================
    if args.step in ("features", "all"):
        print("⚙️ Building daily features (hourly → daily + merge)...")

        build_features(
            weather_hourly_path=args.weather_hourly_path,
            flood_daily_path=args.flood_daily_path,
            out_path=args.features_out_path,
            verbose=True
        )

        if not Path(args.features_out_path).exists():
            raise FileNotFoundError(
                f"❌ Feature file not created: {args.features_out_path}"
            )

    # ============================================================
    # TRAIN
    # ============================================================
    if args.step in ("train", "all"):
        print("🤖 Training CNN-LSTM model...")

        subprocess.run(
            [sys.executable, "model_cnn_lstm.py"],
            check=True
        )

    # ============================================================
    # 🔥 NEW: PREDICTION STEP
    # ============================================================
    if args.step in ("predict", "all"):
        print("🌊 Running flood potential prediction...")

        subprocess.run(
            [sys.executable, "flood-potential-prediction.py"],
            check=True
        )

    print("✅ Done. For EDA + dashboard, run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()