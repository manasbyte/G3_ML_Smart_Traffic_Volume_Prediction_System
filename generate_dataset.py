from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
CSV_PATH = OUT / "traffic_dataset.csv"
EXCEL_PATH = OUT / "traffic_database.xlsx"

def generate_dataset(rows_per_road=1800, seed=42):
    rng = np.random.default_rng(seed)
    roads = {
        "Road A": {"base": 430, "capacity": 1500},
        "Road B": {"base": 320, "capacity": 1200},
        "Road C": {"base": 250, "capacity": 950},
        "Road D": {"base": 520, "capacity": 1700},
    }
    rows = []
    start = pd.Timestamp("2025-01-01 00:00:00")
    total = rows_per_road * len(roads)
    timestamps = start + pd.to_timedelta(rng.integers(0, 365 * 24, size=total), unit="h")
    road_names = list(roads.keys())
    chosen_roads = rng.choice(road_names, size=total)
    for ts, road in zip(timestamps, chosen_roads):
        hour = int(ts.hour)
        day_num = int(ts.dayofweek)
        weekend = day_num >= 5
        holiday = int(rng.random() < 0.035)
        weather = rng.choice(["Clear", "Cloudy", "Rainy", "Foggy"], p=[0.52, 0.25, 0.17, 0.06])
        base = roads[road]["base"]
        # Rush-hour peaks and day-specific patterns
        morning_peak = 420 * np.exp(-((hour - 8) / 2.0) ** 2)
        evening_peak = 560 * np.exp(-((hour - 18) / 2.5) ** 2)
        lunch_peak = 120 * np.exp(-((hour - 13) / 3.0) ** 2)
        day_factor = 1.0 if not weekend else 0.68
        if day_num == 4:
            day_factor *= 1.12
        weather_factor = {"Clear": 1.0, "Cloudy": 0.96, "Rainy": 0.78, "Foggy": 0.72}[weather]
        holiday_factor = 0.70 if holiday else 1.0
        traffic = (base + morning_peak + evening_peak + lunch_peak) * day_factor * weather_factor * holiday_factor
        traffic += rng.normal(0, 65)
        traffic = int(np.clip(traffic, 20, roads[road]["capacity"]))
        rows.append({
            "Timestamp": ts,
            "Road": road,
            "Hour": hour,
            "Day": ts.day_name(),
            "Day_Number": day_num,
            "Is_Weekend": int(weekend),
            "Weather": weather,
            "Is_Holiday": holiday,
            "Temperature_C": int(rng.integers(12, 39)),
            "Traffic_Volume": traffic,
        })
    df = pd.DataFrame(rows).sort_values("Timestamp").reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Traffic_Data")
        summary = df.groupby(["Road", "Hour"], as_index=False)["Traffic_Volume"].mean()
        summary.to_excel(writer, index=False, sheet_name="Road_Hour_Summary")
        pd.DataFrame([
            {"Field": "Timestamp", "Description": "Synthetic date and hour"},
            {"Field": "Road", "Description": "Simulated road identifier"},
            {"Field": "Hour", "Description": "Hour of day, 0-23"},
            {"Field": "Day", "Description": "Day name"},
            {"Field": "Day_Number", "Description": "Monday=0 through Sunday=6"},
            {"Field": "Is_Weekend", "Description": "1 if Saturday/Sunday, else 0"},
            {"Field": "Weather", "Description": "Synthetic weather category"},
            {"Field": "Is_Holiday", "Description": "1 if simulated holiday, else 0"},
            {"Field": "Temperature_C", "Description": "Synthetic temperature in Celsius"},
            {"Field": "Traffic_Volume", "Description": "Simulated vehicle count per observation period"},
        ]).to_excel(writer, index=False, sheet_name="Data_Dictionary")
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Created {len(df):,} rows")
    print(f"CSV: {CSV_PATH}")
    print(f"Excel database: {EXCEL_PATH}")
