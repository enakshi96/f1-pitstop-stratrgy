import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load and prepare dataset
df = pd.read_csv("data/enhanced_strategy_dataset.csv")
df[['Compound2', 'Compound3']] = df[['Compound2', 'Compound3']].fillna('NONE')

# Encode compounds
compound_encoder = LabelEncoder()
all_compounds = pd.concat([df['Compound1'], df['Compound2'], df['Compound3']])
compound_encoder.fit(all_compounds)

df['Compound1_enc'] = compound_encoder.transform(df['Compound1'])
df['Compound2_enc'] = compound_encoder.transform(df['Compound2'])
df['Compound3_enc'] = compound_encoder.transform(df['Compound3'])

# Encode strategy type
strategy_encoder = LabelEncoder()
df['StrategyType_enc'] = strategy_encoder.fit_transform(df['StrategyType'])

# Feature engineering
feature_cols = [
    'QualiPos', 'AvgAirTemp', 'Rain', 'Humidity', 'WindSpeed',
    'DegradationRate', 'TrackLength_km', 'PitLossTime', 'IsStreetCircuit',
    'StrategyType_enc'  # Include as input
]

df = pd.get_dummies(df, columns=['Team', 'Track'])
feature_cols += [col for col in df.columns if col.startswith('Team_') or col.startswith('Track_')]

X = df[feature_cols]
y = df[['Compound1_enc', 'Compound2_enc', 'Compound3_enc']]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
clf.fit(X_train, y_train)

# Evaluate
accs = [clf.estimators_[i].score(X_test, y_test.iloc[:, i]) for i in range(3)]
print(f"📊 Compound1 Accuracy: {accs[0]:.4f}")
print(f"📊 Compound2 Accuracy: {accs[1]:.4f}")
print(f"📊 Compound3 Accuracy: {accs[2]:.4f}")

# Save
os.makedirs("model/encoders", exist_ok=True)
joblib.dump(clf, "model/compound_model.pkl")
joblib.dump(compound_encoder, "model/encoders/compound_encoder.pkl")
joblib.dump(feature_cols, "model/encoders/features_compound.pkl")
