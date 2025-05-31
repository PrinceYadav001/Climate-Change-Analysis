# filepath: [train_aqi_model.py](http://_vscodecontentref_/1)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('india-climate-ml-pbi\data\india_climate_data.csv')
data = data.ffill()  # Fill missing values if any

# Encode categorical columns and save encoders
encoders = {}
categorical_cols = data.select_dtypes(include=['object']).columns.drop(['Record ID', 'Date'])
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le
joblib.dump(encoders, 'aqi_label_encoders.pkl')  # Save all encoders for future use

# Prepare features and target
X = data.drop(['Air Quality Index (AQI)', 'Extreme Weather Events', 'Record ID', 'Date'], axis=1)
y = data['Air Quality Index (AQI)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with tuned parameters
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error for AQI: {mae:.2f}')

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
print("Feature Importances:")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")

# Save the model
joblib.dump(model, 'aqi_model.pkl')