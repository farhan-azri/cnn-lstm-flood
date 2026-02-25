import os

from extract_api import extract_weather_data, extract_flood_data
from eda import run_eda
from feature_engineering import build_features
from model_cnn_lstm import train_model


def install_requirements():
    os.system("pip install -r requirements.txt")


def main():

    print("Installing requirements...")
    install_requirements()

    print("Extracting data...")
    extract_weather_data()
    extract_flood_data()

    print("Running EDA...")
    run_eda()

    print("Building Features...")
    build_features()

    print("Training Model...")
    train_model()

    print("Done!")


if __name__ == "__main__":
    main()