import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load data
df = pd.read_csv("data/enhanced_strategy_dataset.csv")

# Filter valid strategy types
valid_strategies = ['1-stop', '2-stop', '3-stop']
df = df[df['StrategyType'].isin(valid_strategies)]

# Encode target
strategy_encoder = LabelEncoder()
df['StrategyType_enc'] = strategy_encoder.fit_transform(df['StrategyType'])

# Define features
feature_cols = [
    'QualiPos', 'AvgAirTemp', 'Rain', 'Humidity', 'WindSpeed',
    'DegradationRate', 'TrackLength_km', 'PitLossTime', 'IsStreetCircuit'
]

df = pd.get_dummies(df, columns=['Team', 'Track'])
feature_cols += [col for col in df.columns if col.startswith('Team_') or col.startswith('Track_')]

X = df[feature_cols]
y = df['StrategyType_enc']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate
acc = clf.score(X_test, y_test)
print(f"✅ Strategy Type Model Accuracy: {acc:.4f}")

# Save model and encoders
os.makedirs("model/encoders", exist_ok=True)
joblib.dump(clf, "model/strategy_model.pkl")
joblib.dump(strategy_encoder, "model/encoders/strategy_encoder.pkl")
joblib.dump(feature_cols, "model/encoders/features_strategy.pkl")
