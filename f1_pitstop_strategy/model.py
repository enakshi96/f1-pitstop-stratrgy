import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load and clean dataset
df = pd.read_csv('data/enhanced_strategy_dataset.csv')
valid_strategies = ['2-stop']
df = df[df['StrategyType'].isin(valid_strategies)]
df[['Compound2', 'Compound3']] = df[['Compound2', 'Compound3']].fillna('NONE')

# Encode compounds and strategy type
compound_encoder = LabelEncoder()
all_compounds = pd.concat([df['Compound1'], df['Compound2'], df['Compound3']])
compound_encoder.fit(all_compounds)

df['Compound1_enc'] = compound_encoder.transform(df['Compound1'])
df['Compound2_enc'] = compound_encoder.transform(df['Compound2'])
df['Compound3_enc'] = compound_encoder.transform(df['Compound3'])

strategy_encoder = LabelEncoder()
df['StrategyType_enc'] = strategy_encoder.fit_transform(df['StrategyType'])

# Create tire performance features
pivot = df.pivot_table(
    index=['Team', 'Track'],
    columns='Compound1',
    values='Stint1AvgTime',
    aggfunc='mean'
).reset_index()
pivot.columns = ['Team', 'Track'] + [f'Tire_{c}_Perf' for c in pivot.columns[2:]]
df = df.merge(pivot, on=['Team', 'Track'], how='left')
df.fillna(0, inplace=True)  # Handle missing performance features

# Feature selection
feature_cols = ['QualiPos', 'Team', 'Track', 'AvgAirTemp', 'Rain', 'Humidity',
                'WindSpeed', 'DegradationRate', 'TrackLength_km', 'PitLossTime',
                'IsStreetCircuit'] + [col for col in df.columns if col.startswith('Tire_')]

df = pd.get_dummies(df, columns=['Team', 'Track'])

X = df[[col for col in df.columns if col in feature_cols or col.startswith('Team_') or col.startswith('Track_')]]

# Targets
y_cls = df[['StrategyType_enc', 'Compound1_enc', 'Compound2_enc', 'Compound3_enc']]
y_reg = df[['Stint1Laps', 'Stint2Laps', 'Stint3Laps']]

# Train/test split
X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
    X, y_cls, y_reg, test_size=0.2, random_state=42
)

# Models
clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
clf.fit(X_train, y_cls_train)

reg = MultiOutputRegressor(RandomForestRegressor(random_state=42))
reg.fit(X_train, y_reg_train)

# Evaluate
print("📊 Classification Accuracy:", clf.score(X_test, y_cls_test))
print("📊 Regression R² Score:", reg.score(X_test, y_reg_test))

# Save everything
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/classifier.pkl")
joblib.dump(reg, "model/regressor.pkl")
joblib.dump(compound_encoder, "model/compound_label_encoder.pkl")
joblib.dump(strategy_encoder, "model/strategy_label_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/model_features.pkl")

print("✅ Model training complete and files saved.")

