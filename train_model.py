import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

np.random.seed(42)
districts = ["central", "north", "south", "west", "east"]
building_types = ["panel", "brick", "monolith"]
conditions = ["excellent", "good", "needs_repair"]

data = {
    "district": np.random.choice(districts, 1000),
    "rooms": np.random.randint(1, 5, 1000),
    "square": np.round(np.random.uniform(30, 120, 1000), 1),
    "floor": np.random.randint(1, 20, 1000),
    "total_floors": np.random.randint(5, 25, 1000),
    "building_type": np.random.choice(building_types, 1000),
    "condition": np.random.choice(conditions, 1000),
    "price": np.random.randint(50_000, 500_000, 1000)
}

df = pd.DataFrame(data)

df['room_size'] = df['square'] / df['rooms']
df['floor_ratio'] = df['floor'] / df['total_floors']
df['is_top_floor'] = (df['floor'] == df['total_floors']).astype(int)

for col in ["district", "building_type", "condition"]:
    df[col] = df[col].astype('category').cat.codes

X = df.drop('price', axis=1)
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")