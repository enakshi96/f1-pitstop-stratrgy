from fastf1 import get_session, Cache
import pandas as pd
from analyse import generate_insights

# Enable FastF1 cache
Cache.enable_cache('f1_pit_cache')
def analyze_strategy(year: int, race_name: str, driver_code: str):
    from fastf1 import get_session
    from fastf1.core import DataNotLoadedError

    try:
        session = get_session(year, race_name, 'R')
        session.load()
    except Exception as e:
        return {"error": f"Failed to load race data: {str(e)}"}

    try:
        laps = session.laps.pick_driver(driver_code)
    except DataNotLoadedError:
        return {"error": "Session data not available yet. The race may not have occurred or data is not published."}

    if laps.empty:
        return {"error": "No lap data found for this driver."}

    try:
        session = get_session(year, race_name, 'R')
        session.load()

    except Exception as e:
        return {"error": f"Failed to load race: {str(e)}"}
    laps = session.laps.pick_driver(driver_code)
    for drv in session.drivers:
        info = session.get_driver(drv)
        print(f"{info['Abbreviation']} - {info['FullName']}")

    if laps.empty:
        return {"error": "No lap data found for this driver."}
    stints = laps[['LapNumber', 'Stint', 'Compound']]
    pit_stops = laps[laps['PitInTime'].notna()]['LapNumber'].tolist()
    strategy_type = f"{len(pit_stops)}-stop" if len(pit_stops) <= 3 else "multi-stop"

    # Build stint summary
    stint_summary = []
    for stint_number in stints['Stint'].unique():
        stint_data = stints[stints['Stint'] == stint_number]
        compound = stint_data['Compound'].iloc[0]
        length = len(stint_data)
        stint_summary.append({
            "stint": int(stint_number),
            "compound": compound,
            "length": length
        })

   

    # Lap times (optional for plotting)
    lap_times = laps[['LapNumber', 'LapTime']].dropna()
    lap_times['LapTime'] = lap_times['LapTime'].dt.total_seconds()

    strategy_data = {
        "strategy": strategy_type,
        "pit_stops": pit_stops,
        "stints": stint_summary
    }

    try:
        insight = generate_insights(strategy_data)
    except Exception as e:
        insight = f"Insight generation failed: {str(e)}"

    return {
        "strategy": strategy_type,
        "pit_stops": pit_stops,
        "stints": stint_summary,
        "lap_times": lap_times.to_dict(orient='records'),
        "insight": insight
    }

   