import requests
from datetime import datetime
race_to_location = {
    "British Grand Prix": "Silverstone",
    "Monaco Grand Prix": "Monaco",
    "Japanese Grand Prix": "Suzuka",
    "Italian Grand Prix": "Monza",
    "Bahrain Grand Prix": "Sakhir",
    "Miami Grand Prix": "Miami",
    "Canadian Grand Prix": "Montreal",
    "United States Grand Prix": "Austin",
    "Abu Dhabi Grand Prix": "Abu Dhabi",
    "Singapore Grand Prix": "Singapore",
    "Spanish Grand Prix": "Barcelona",
    "Hungarian Grand Prix": "Budapest",
    "Dutch Grand Prix": "Zandvoort",
    "Australian Grand Prix": "Melbourne",
    # Add more mappings as needed
}

track_coordinates = {
    "Silverstone": (52.0786, -1.0169),
    "Monaco": (43.7384, 7.4246),
    "Suzuka": (34.8431, 136.5419),
    "Monza": (45.6156, 9.2811),
    "Sakhir": (26.0325, 50.5106),
    "Miami": (25.7617, -80.1918),
    "Montreal": (45.5017, -73.5673),
    "Austin": (30.2672, -97.7431),
    "Abu Dhabi": (24.4539, 54.3773),
    "Singapore": (1.3521, 103.8198),
    "Barcelona": (41.3851, 2.1734),
    "Budapest": (47.4979, 19.0402),
    "Zandvoort": (52.3740, 4.5331),
    "Melbourne": (-37.8136, 144.9631),
    # Add more as needed
}


def get_live_weather(track: str, date: str, time: str):
    location = race_to_location.get(track)
    if not location:
        print(f"⚠️ Unknown location for race: {track}")
        return {
            "AvgAirTemp": 25.0,
            "Rain": 0.0,
            "WindSpeed": 0.0,
            "Humidity": 50.0
        }
    lat, lon = track_coordinates.get(location, (0, 0))
    target_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=relative_humidity_2m,precipitation, wind_speed_10m"
        f"&timezone=auto"
    )

    try:
        res = requests.get(url)
        data = res.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            raise ValueError("No hourly forecast data")

        time_objs = [datetime.fromisoformat(t) for t in times]
        idx = min(range(len(time_objs)), key=lambda i: abs(time_objs[i] - target_dt))

        return {
            "AvgAirTemp": hourly["temperature_2m"][idx],
            "Humidity": hourly["relative_humidity_2m"][idx],
            "Rain": hourly["precipitation"][idx],
            "WindSpeed": hourly["wind_speed_10m"][idx]
        }

    except Exception as e:
        print(f"⚠️ Weather fetch failed for {track}: {e}")
        return {
            "AvgAirTemp": 25.0,
            "Rain": 0.0,
            "WindSpeed": 0.0,
            "Humidity": 50.0
        }
