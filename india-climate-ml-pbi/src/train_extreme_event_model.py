import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv('india-climate-ml-pbi\data\india_climate_data.csv')
data = data.ffill()

# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.drop(['Record ID', 'Date', 'Extreme Weather Events'])
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col].astype(str))

# ADD THIS LINE:
data['Extreme Weather Events'] = data['Extreme Weather Events'].fillna('None')
# ...rest of your code...
le = LabelEncoder()
data['Extreme Weather Events'] = le.fit_transform(data['Extreme Weather Events'].astype(str))
joblib.dump(le, 'event_label_encoder.pkl')


# Prepare features and target
X = data.drop(['Air Quality Index (AQI)', 'Extreme Weather Events', 'Record ID', 'Date'], axis=1)
y = data['Extreme Weather Events']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model with class_weight balanced
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f'Accuracy for Extreme Weather Events: {acc:.2f}')
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Save the model
joblib.dump(model, 'extreme_event_model.pkl')