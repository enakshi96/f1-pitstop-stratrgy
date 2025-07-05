import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import joblib
import os

# Load dataset
df = pd.read_csv("data/enhanced_strategy_dataset.csv")
df[['Compound2', 'Compound3']] = df[['Compound2', 'Compound3']].fillna('NONE')

# Encode strategy type
strategy_encoder = LabelEncoder()
df['StrategyType_enc'] = strategy_encoder.fit_transform(df['StrategyType'])

# Encode compounds
compound_encoder = LabelEncoder()
all_compounds = pd.concat([df['Compound1'], df['Compound2'], df['Compound3']])
compound_encoder.fit(all_compounds)

df['Compound1_enc'] = compound_encoder.transform(df['Compound1'])
df['Compound2_enc'] = compound_encoder.transform(df['Compound2'])
df['Compound3_enc'] = compound_encoder.transform(df['Compound3'])

# Features
feature_cols = [
    'QualiPos', 'AvgAirTemp', 'Rain', 'Humidity', 'WindSpeed',
    'DegradationRate', 'TrackLength_km', 'PitLossTime', 'IsStreetCircuit',
    'StrategyType_enc', 'Compound1_enc', 'Compound2_enc', 'Compound3_enc'
]

df = pd.get_dummies(df, columns=['Team', 'Track'])
feature_cols += [col for col in df.columns if col.startswith('Team_') or col.startswith('Track_')]

# Features, target, total laps
X = df[feature_cols]
y = df[['Stint1Laps', 'Stint2Laps', 'Stint3Laps']]
laps = df['TotalLaps']

# Train-test split
X_train, X_test, y_train, y_test, laps_train, laps_test = train_test_split(
    X, y, laps, test_size=0.2, random_state=42
)

# Train model
reg = MultiOutputRegressor(RandomForestRegressor(random_state=42))
reg.fit(X_train, y_train)

# Predict raw
y_pred_raw = reg.predict(X_test)

# ✅ Scale predictions to match total laps
def scale_to_laps(pred_stints, total_laps_list):
    scaled_preds = []
    for pred, total_laps in zip(pred_stints, total_laps_list):
        pred_sum = sum(pred)
        if pred_sum == 0:
            scaled = [0, 0, total_laps]
        else:
            scaled = [round(s * total_laps / pred_sum) for s in pred]
            diff = total_laps - sum(scaled)
            scaled[-1] += diff
        scaled_preds.append(scaled)
    return pd.DataFrame(scaled_preds, columns=y.columns)

# Apply scaling
y_pred_scaled = scale_to_laps(y_pred_raw, laps_test.values)

# R² evaluation after scaling
print("📊 R² Scores (After Scaling to TotalLaps):")
for i, col in enumerate(y.columns):
    r2 = r2_score(y_test.iloc[:, i], y_pred_scaled.iloc[:, i])
    print(f"   {col}: {r2:.4f}")

# Save model
os.makedirs("model/encoders", exist_ok=True)
joblib.dump(reg, "model/stint_model.pkl")
joblib.dump(feature_cols, "model/encoders/features_stint.pkl")
