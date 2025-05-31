import pandas as pd
import joblib

# Load the dataset
data = pd.read_csv('india-climate-ml-pbi\data\india_climate_data.csv')
data = data.ffill()  # Fill missing values if any

# Replace NaN in event column with 'None'
data['Extreme Weather Events'] = data['Extreme Weather Events'].fillna('None')

# Load encoders used during training
encoders = joblib.load('aqi_label_encoders.pkl')
event_le = joblib.load('event_label_encoder.pkl')

# Encode categorical columns using saved encoders
categorical_cols = data.select_dtypes(include=['object']).columns.drop(['Record ID', 'Date', 'Extreme Weather Events'])
for col in categorical_cols:
    le = encoders[col]
    data[col] = le.transform(data[col].astype(str))

X = data.drop(['Air Quality Index (AQI)', 'Extreme Weather Events', 'Record ID', 'Date'], axis=1)

# Load models
aqi_model = joblib.load('aqi_model.pkl')
event_model = joblib.load('extreme_event_model.pkl')

# Predict
data['Predicted_AQI'] = aqi_model.predict(X)
data['Predicted_Event'] = event_model.predict(X)
data['Predicted_Event'] = event_le.inverse_transform(data['Predicted_Event'])

# Add actual event labels (decode if needed)
data['Actual_Event'] = event_le.inverse_transform(
    event_le.transform(data['Extreme Weather Events'].astype(str))
)

# Export for Power BI (with actuals)
data[['Date', 'City', 'Air Quality Index (AQI)', 'Actual_Event', 'Predicted_AQI', 'Predicted_Event']].to_csv('predictions_for_powerbi.csv', index=False)
print("Exported predictions_for_powerbi.csv")