import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda():
    weather = pd.read_csv("data/weather.csv")
    flood = pd.read_csv("data/flood.csv")

    weather["date"] = pd.to_datetime(weather["date"])
    flood["date"] = pd.to_datetime(flood["date"])

    sns.histplot(weather["rain_sum_mm"], kde=True)
    plt.title("Rainfall Distribution")
    plt.show()

    sns.lineplot(data=flood, x="date", y="river_discharge", hue="location")
    plt.title("River Discharge Trend")
    plt.show()