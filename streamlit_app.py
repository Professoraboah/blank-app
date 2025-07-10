# Install dependencies
!pip install pandas numpy scikit-learn xgboost tensorflow joblib

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Cities
cities = {
    "Accra": (5.56, -0.20),
    "Kumasi": (6.69, -1.62),
    "Cape Coast": (5.11, -1.25),
    "Takoradi": (4.89, -1.75)
}

# Parameters
features = ['temperature', 'humidity', 'wind_speed', 'wind_dir', 'pm10', 'o3', 'no2', 'so2', 'co']
n_points = 60
hours = list(range(24))
data = []

np.random.seed(42)
for city, (lat, lon) in cities.items():
    for gid in range(n_points):
        for hr in hours:
            data.append({
                'city': city,
                'grid_id': f"{city[:2]}_{gid}", 'hour': hr,
                'lat': lat + np.random.uniform(-0.02, 0.02),
                'lon': lon + np.random.uniform(-0.02, 0.02),
                'temperature': 25 + np.random.randn()*2,
                'humidity': 70 + np.random.randn()*10,
                'wind_speed': 1 + np.random.rand()*4,
                'wind_dir': np.random.rand()*360,
                'pm25': np.random.rand()*150,
                'pm10': np.random.rand()*180,
                'o3': np.random.rand()*120,
                'no2': np.random.rand()*80,
                'so2': np.random.rand()*60,
                'co': np.random.rand()*15,
            })

df = pd.DataFrame(data)
df.to_pickle("ghana_air_quality_dataset.pkl")

# Train models
X = df[features]
y = df['pm25']
joblib.dump(RandomForestRegressor(n_estimators=50).fit(X, y), "rf_pm25_model.joblib")
joblib.dump(XGBRegressor(n_estimators=50).fit(X, y), "xgb_pm25_model.joblib")

# LSTM
lstm_X, lstm_y = [], []
for gid in df['grid_id'].unique():
    subset = df[df['grid_id'] == gid].sort_values('hour')
    if len(subset) == 24:
        lstm_X.append(subset[features].values)
        lstm_y.append(subset['pm25'].values[-1])
lstm_X, lstm_y = np.array(lstm_X), np.array(lstm_y)

model = Sequential([
    LSTM(32, input_shape=(24, len(features))),
    Dense(1)
])
model.compile(optimizer=Adam(0.001), loss='mse')
model.fit(lstm_X, lstm_y, epochs=5, batch_size=8, verbose=1)
model.save("lstm_pm25_model.h5")
