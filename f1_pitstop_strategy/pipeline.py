import pandas as pd
from track_data import track_info

# Load models and encoders
import joblib

strategy_type_model = joblib.load("model/strategy_model.pkl")
compound_model = joblib.load("model/compound_model.pkl")
stint_model = joblib.load("model/stint_model.pkl")

strategy_encoder = joblib.load("model/strategy_label_encoder.pkl")
compound_encoder = joblib.load("model/compound_label_encoder.pkl")

features_strategy = joblib.load("model/encoders/features_strategy.pkl")
features_compound = joblib.load("model/encoders/features_compound.pkl")
features_stint = joblib.load("model/encoders/features_stint.pkl")


def unified_predict_pipeline(input_dict: dict, track: str) -> dict:
    """
    Unified pipeline: predicts strategy type, compounds, and stint lengths
    using the trained models and postprocessing to match TotalLaps.
    """
    # --- Strategy Prediction ---
    X_strategy = pd.DataFrame([input_dict])[features_strategy]
    strategy_pred = strategy_type_model.predict(X_strategy)[0]
    strategy_type = strategy_encoder.inverse_transform([strategy_pred])[0]
    input_dict["StrategyType_enc"] = strategy_pred

    # --- Compound Prediction ---
    X_compound = pd.DataFrame([input_dict])[features_compound]
    compound_preds = compound_model.predict(X_compound)[0]
    compounds = compound_encoder.inverse_transform(compound_preds)
    input_dict["Compound1_enc"] = compound_preds[0]
    input_dict["Compound2_enc"] = compound_preds[1]
    input_dict["Compound3_enc"] = compound_preds[2]

    # --- Stint Length Prediction ---
    X_stint = pd.DataFrame([input_dict])[features_stint]
    stint_preds = stint_model.predict(X_stint)[0]

    # --- Postprocessing to match TotalLaps ---
    short_track = track.split()[0]  # e.g., 'British' from 'British Grand Prix'
    total_laps = track_info.get(short_track, {}).get("TotalLaps", 52)  # fallback = 52


    def scale_to_laps(pred, total):
        pred_sum = sum(pred)
        if pred_sum == 0:
            return [0, 0, total]
        scaled = [round(p * total / pred_sum) for p in pred]
        scaled[-1] += total - sum(scaled)  # fix rounding mismatch
        return scaled

    scaled_stints = scale_to_laps(stint_preds, total_laps)

    # --- Final Output ---
    return {
        "strategy_type": strategy_type,
        "compounds": {
            "Stint 1": compounds[0],
            "Stint 2": compounds[1],
            "Stint 3": compounds[2],
        },
        "stint_lengths": {
            "Stint 1": int(scaled_stints[0]),
            "Stint 2": int(scaled_stints[1]),
            "Stint 3": int(scaled_stints[2]),
        }
    }
