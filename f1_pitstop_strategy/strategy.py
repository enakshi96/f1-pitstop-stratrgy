import fastf1
from fastf1 import Cache
import pandas as pd
import os

# Enable FastF1 cache
Cache.enable_cache('f1_cache')
track_lengths_km = {
    "Bahrain Grand Prix": 5.412,
    "Saudi Arabian Grand Prix": 6.174,
    "Australian Grand Prix": 5.278,
    "Emilia Romagna Grand Prix": 4.909,
    "Miami Grand Prix": 5.412,
    "Spanish Grand Prix": 4.675,
    "Monaco Grand Prix": 3.337,
    "Azerbaijan Grand Prix": 6.003,
    "Canadian Grand Prix": 4.361,
    "British Grand Prix": 5.891,
    "Austrian Grand Prix": 4.318,
    "Hungarian Grand Prix": 4.381,
    "Belgian Grand Prix": 7.004,
    "Dutch Grand Prix": 4.259,
    "Italian Grand Prix": 5.793,
    "Singapore Grand Prix": 4.940,
    "Japanese Grand Prix": 5.807,
    "Qatar Grand Prix": 5.380,
    "United States Grand Prix": 5.513,
    "Mexico City Grand Prix": 4.304,
    "São Paulo Grand Prix": 4.309,
    "Las Vegas Grand Prix": 6.201,
    "Abu Dhabi Grand Prix": 5.281
}
# Manual track info (for missing fields)
track_info = {
    "Monza": {"PitLossTime": 21.0, "IsStreetCircuit": 0},
    "Monaco": {"PitLossTime": 22.5, "IsStreetCircuit": 1},
    "Silverstone": {"PitLossTime": 20.0, "IsStreetCircuit": 0},
    "Hungaroring": {"PitLossTime": 19.8, "IsStreetCircuit": 0},
    "Spa": {"PitLossTime": 23.0, "IsStreetCircuit": 0},
    "Suzuka": {"PitLossTime": 20.2, "IsStreetCircuit": 0},
    "Bahrain": {"PitLossTime": 20.5, "IsStreetCircuit": 0},
    "Shanghai": {"PitLossTime": 21.5, "IsStreetCircuit": 0},
    "Las": {"PitLossTime": 22.0, "IsStreetCircuit": 1},
    "Qatar": {"PitLossTime": 20.0, "IsStreetCircuit": 0},
    "Jeddah": {"PitLossTime": 21.5, "IsStreetCircuit": 1},
    "Miami": {"PitLossTime": 21.0, "IsStreetCircuit": 1}
}

# Expand races list for 2022-2024 seasons
races = []
seasons = [2022, 2023, 2024]

events_per_season = {
    2022: [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Azerbaijan Grand Prix", "Miami Grand Prix", "Monaco Grand Prix",
        "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix",
        "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix",
        "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix",
        "Japanese Grand Prix", "Qatar Grand Prix", "United States Grand Prix",
        "Mexican Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix",
        "Abu Dhabi Grand Prix"
    ],
    2023: [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Azerbaijan Grand Prix", "Miami Grand Prix", "Monaco Grand Prix",
        "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix",
        "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix",
        "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix",
        "Japanese Grand Prix", "Qatar Grand Prix", "United States Grand Prix",
        "Mexican Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix",
        "Abu Dhabi Grand Prix"
    ],
    2024: [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
        "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
        "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
        "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
        "Italian Grand Prix", "Singapore Grand Prix", "United States Grand Prix",
        "Mexico Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix",
        "Qatar Grand Prix", "Abu Dhabi Grand Prix"
    ]
}

for season in seasons:
    for race in events_per_season[season]:
        races.append((season, race))

# Data collection
rows = []

for year, race_name in races:
    try:
        session = fastf1.get_session(year, race_name, "R")
        session.load()
        short_name = race_name.split()[0]
        meta = track_info.get(short_name, {"PitLossTime": 20.0, "IsStreetCircuit": 0})

        # Track length from session
        track_name = session.event['EventName']
        total_laps = session.total_laps
        track_length = track_lengths_km.get(track_name, None) / session.total_laps
        weather = session.weather_data
        avg_temp = weather['AirTemp'].mean()
        rain = int(weather['Rainfall'].max() > 0)
        humidty = weather['Humidity'].mean()
        wind_speed = weather['WindSpeed'].mean()

        for drv_id in session.drivers:
            try:
                drv_info = session.get_driver(drv_id)
                driver_code = drv_info['Abbreviation']
                team = drv_info['TeamName']
                quali_pos = session.results.loc[drv_id]['GridPosition']

                laps = session.laps.pick_driver(driver_code).dropna(subset=['LapTime'])
                if laps.empty:
                    continue
                driver_laps_completed = laps['LapNumber'].nunique()
                if driver_laps_completed < 0.9 * session.total_laps:
                    continue
                laps['LapTime'] = laps['LapTime'].dt.total_seconds()
                avg_lap_time = laps['LapTime'].mean()
                laps['LapTimeDiff'] = laps['LapTime'].diff().fillna(0)
                degradation = laps['LapTimeDiff'].mean()
                sc_laps = laps[laps['TrackStatus'].astype(str).str.contains("4")]['LapNumber'].nunique()
                overtakes = (laps['Position'].diff().fillna(0) < 0).sum()

                pit_laps = laps[laps['PitInTime'].notna()]
                strategy_type = f"{len(pit_laps)}-stop" if len(pit_laps) <= 3 else "multi-stop"

                stints_data = laps[['Stint', 'Compound', 'LapNumber']].groupby('Stint')
                stints = []
                for _, group in stints_data:
                    compound = group['Compound'].iloc[0]
                    stint_len = len(group)
                    avg_time = laps.loc[group.index, 'LapTime'].mean()
                    stints.append({"compound": compound, "length": stint_len, "avg_time": avg_time})
                while len(stints) < 3:
                    stints.append({"compound": None, "length": 0, "avg_time": 0})

                row = {
                    "Year": year,
                    "Track": race_name,
                    "Driver": driver_code,
                    "Team": team,
                    "QualiPos": quali_pos,
                    "AvgAirTemp": avg_temp,
                    "Rain": rain,
                    "Humidity": humidty,
                    "WindSpeed": wind_speed,
                    "TrackLength_km": track_length,
                    "TotalLaps": total_laps,
                    "PitLossTime": meta["PitLossTime"],
                    "IsStreetCircuit": meta["IsStreetCircuit"],
                    "AvgLapTime": avg_lap_time,
                    "DegradationRate": degradation,
                    "SafetyCarLaps": sc_laps,
                    "OvertakesMade": overtakes,
                    "StrategyType": strategy_type,
                    "Compound1": stints[0]["compound"],
                    "Stint1Laps": stints[0]["length"],
                    "Stint1AvgTime": stints[0]["avg_time"],
                    "Compound2": stints[1]["compound"],
                    "Stint2Laps": stints[1]["length"],
                    "Stint2AvgTime": stints[1]["avg_time"],
                    "Compound3": stints[2]["compound"],
                    "Stint3Laps": stints[2]["length"],
                    "Stint3AvgTime": stints[2]["avg_time"],
                }

                rows.append(row)

            except Exception as e:
                print(f"⚠️ Skipped driver {drv_id} in {race_name}: {e}")
    except Exception as e:
        print(f"❌ Failed to load {race_name}: {e}")

# Save CSV
df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/enhanced_strategy_dataset.csv", index=False)
print("✅ Saved to data/enhanced_strategy_dataset.csv")

   