from nbformat import v4 as nbf
import nbformat
from pathlib import Path

# Define output path for the notebook
notebook_path = Path("end_to_end_forecasting.ipynb")

# Create a new Jupyter notebook
nb = nbf.new_notebook()

# Notebook cells
cells = []

# Title and intro
cells.append(nbf.new_markdown_cell("# End-to-End Time Series Forecasting with RF, LSTM, and Prophet"))
cells.append(nbf.new_markdown_cell("""
This notebook demonstrates a complete time series forecasting pipeline on daily energy prices using:
- Feature-engineered **Random Forest**
- Deep learning with **LSTM**
- Time series modeling with **Prophet**

Each model is evaluated and compared using MAE and RMSE.
"""))

# Imports
cells.append(nbf.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# For LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# For Prophet
from prophet import Prophet
"""))

# Load & preprocess data
cells.append(nbf.new_code_cell("""
# Sample data
data = {
    "Datetime": pd.date_range(start="2018-01-01", periods=1000, freq='D'),
    "MarginalES": np.random.rand(1000) * 100
}
df = pd.DataFrame(data)

# Extract date features
df['dayofweek'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month
df['dayofyear'] = df['Datetime'].dt.dayofyear
df.set_index('Datetime', inplace=True)

# Create lagged features
for lag in range(1, 8):
    df[f"lag_{lag}"] = df["MarginalES"].shift(lag)

df.dropna(inplace=True)
"""))

# Split data
cells.append(nbf.new_code_cell("""
features = [col for col in df.columns if col != "MarginalES"]
X = df[features]
y = df["MarginalES"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
"""))

# RF model
cells.append(nbf.new_code_cell("""
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def evaluate(y_true, y_pred, label="Set"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    print(f"{label} MAE: {mae:.2f}")
    print(f"{label} RMSE: {rmse:.2f}")
    return mae, rmse

print("Random Forest Evaluation")
evaluate(y_val, model.predict(X_val), "Validation")
evaluate(y_test, model.predict(X_test), "Test")
"""))

# LSTM model
cells.append(nbf.new_code_cell("""
# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Prepare LSTM format: [samples, time steps, features]
def create_sequences(X, y, timesteps=7):
    Xs, ys = [], []
    for i in range(timesteps, len(X)):
        Xs.append(X[i-timesteps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_lstm, y_lstm = create_sequences(X_scaled, y.values, timesteps=7)

# Train/val/test
split1 = int(0.8 * len(X_lstm))
split2 = int(0.9 * len(X_lstm))
X_train_lstm, X_val_lstm, X_test_lstm = X_lstm[:split1], X_lstm[split1:split2], X_lstm[split2:]
y_train_lstm, y_val_lstm, y_test_lstm = y_lstm[:split1], y_lstm[split1:split2], y_lstm[split2:]

model_lstm = Sequential([
    LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train_lstm, y_train_lstm, epochs=10, validation_data=(X_val_lstm, y_val_lstm), verbose=1)

print("LSTM Evaluation")
evaluate(y_val_lstm, model_lstm.predict(X_val_lstm).flatten(), "Validation")
evaluate(y_test_lstm, model_lstm.predict(X_test_lstm).flatten(), "Test")
"""))

# Prophet model
cells.append(nbf.new_code_cell("""
# Rebuild Prophet-compatible dataset
df_prophet = df[['MarginalES']].copy().reset_index()
df_prophet.columns = ['ds', 'y']

# Split for Prophet
train_prophet = df_prophet[:-60]
test_prophet = df_prophet[-60:]

model_prophet = Prophet()
model_prophet.fit(train_prophet)

future = model_prophet.make_future_dataframe(periods=60)
forecast = model_prophet.predict(future)

# Evaluate Prophet
pred = forecast[['ds', 'yhat']].set_index('ds').join(test_prophet.set_index('ds'))
evaluate(pred['y'], pred['yhat'], label="Prophet Test")

fig = model_prophet.plot(forecast)
plt.title("Prophet Forecast")
plt.show()
"""))

# Finalize notebook
nb['cells'] = cells
with open(notebook_path, "w") as f:
    nbformat.write(nb, f)

notebook_path.name
