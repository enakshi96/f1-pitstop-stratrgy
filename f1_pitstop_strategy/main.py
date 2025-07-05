from fastapi import FastAPI, Query
import pandas as pd
from datetime import datetime
from weather import get_live_weather
from track_data import track_lengths_km, track_info
from llm_explanation import generate_llm_explanation
from pipeline import unified_predict_pipeline  
from fastapi.responses import HTMLResponse

# Initialize FastAPI app
app = FastAPI()


@app.get("/predict_strategy")
def predict_strategy(
    track: str = Query(..., description="Grand Prix name, e.g., 'British Grand Prix'"),
    team: str = Query(..., description="Team name, e.g., 'Ferrari'"),
    quali_pos: int = Query(..., description="Qualifying position, e.g., 3"),
    date: str = Query(..., description="Race date in YYYY-MM-DD format"),
    time: str = Query(..., description="Race time in HH:MM (24hr format)")
):
    # Step 1: Get live weather data
    weather = get_live_weather(track, date, time)

    # Step 2: Build input vector
    input_data = build_input_vector(track, team, quali_pos, weather)

    # Step 3: Run unified pipeline
    result = unified_predict_pipeline(input_data, track)

    # Step 4: Add metadata

    result["weather"] = weather
    prediction_list = [
    result["strategy_type"],
    result["compounds"]["Stint 1"],
    result["compounds"]["Stint 2"],
    result["compounds"]["Stint 3"],
    result["stint_lengths"]["Stint 1"],
    result["stint_lengths"]["Stint 2"],
    result["stint_lengths"]["Stint 3"],
]
    result["explanation"] = generate_llm_explanation(input_data, prediction_list)

    return result


def build_input_vector(track: str, team: str, quali_pos: int, weather: dict) -> dict:
    from pipeline import features_strategy  # Use one model’s feature list as master list

    short_track = track.split()[0]
    track_length = track_lengths_km.get(track, 5.5)
    pit_info = track_info.get(short_track, {"PitLossTime": 20.0, "IsStreetCircuit": 0})

    base_input = {
        "QualiPos": quali_pos,
        "AvgAirTemp": weather.get("AvgAirTemp", 25),
        "Rain": 1 if weather.get("Rain", 0) > 0 else 0,
        "Humidity": weather.get("Humidity", 50),
        "WindSpeed": weather.get("WindSpeed", 5),
        "TrackLength_km": track_length,
        "PitLossTime": pit_info["PitLossTime"],
        "IsStreetCircuit": pit_info["IsStreetCircuit"],
        "DegradationRate": 0.15
    }

    # Add all expected one-hot columns with default 0
    for col in features_strategy:
        if col.startswith("Team_") or col.startswith("Track_"):
            base_input[col] = 1 if col == f"Team_{team}" or col == f"Track_{track}" else 0

    return base_input

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>F1 Pit Strategy Predictor</title>
        <style>
            body { font-family: sans-serif; padding: 20px; background: #111; color: #eee; }
            h1 { color: #f33; }
            label, input, select { display: block; margin: 10px 0; }
            button { padding: 10px 20px; background: #f33; color: white; border: none; }
            .result { margin-top: 20px; background: #222; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>F1 Pit Strategy Predictor</h1>
        <form id="strategyForm">
            <label>Track: <input type="text" id="track" value="British Grand Prix" required></label>
            <label>Team: <input type="text" id="team" value="Ferrari" required></label>
            <label>Qualifying Position: <input type="number" id="quali" value="3" required></label>
            <label>Date (YYYY-MM-DD): <input type="text" id="date" value="2025-08-04" required></label>
            <label>Time (HH:MM): <input type="text" id="time" value="14:30" required></label>
            <button type="submit">Predict Strategy</button>
        </form>
        <div class="result" id="result"></div>

        <script>
            const form = document.getElementById('strategyForm');
            const resultDiv = document.getElementById('result');

            form.onsubmit = async (e) => {
                e.preventDefault();
                const track = document.getElementById('track').value;
                const team = document.getElementById('team').value;
                const quali = document.getElementById('quali').value;
                const date = document.getElementById('date').value;
                const time = document.getElementById('time').value;

                const res = await fetch(`/predict_strategy?track=${track}&team=${team}&quali_pos=${quali}&date=${date}&time=${time}`);
                const data = await res.json();

                resultDiv.innerHTML = `
                    <h3>🏁 Prediction</h3>
                    <p><b>Strategy Type:</b> ${data.strategy_type}</p>
                    <p><b>Compounds:</b><br>
                        Stint 1: ${data.compounds["Stint 1"]}<br>
                        Stint 2: ${data.compounds["Stint 2"]}<br>
                        Stint 3: ${data.compounds["Stint 3"]}
                    </p>
                    <p><b>Stint Lengths:</b><br>
                        Stint 1: ${data.stint_lengths["Stint 1"]} laps<br>
                        Stint 2: ${data.stint_lengths["Stint 2"]} laps<br>
                        Stint 3: ${data.stint_lengths["Stint 3"]} laps
                    </p>
                    <p><b>Weather:</b> ${JSON.stringify(data.weather)}</p>
                    <p><b>Explanation:</b> ${data.explanation}</p>
                `;
            };
        </script>
    </body>
    </html>
    """