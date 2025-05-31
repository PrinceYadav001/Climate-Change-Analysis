import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report

def main():
    # Load test data (use forward slash for cross-platform compatibility)
    data = pd.read_csv('india-climate-ml-pbi\data\india_climate_test_data.csv')
    data = data.ffill()  # Fill missing values

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

    # Encode Extreme Weather Events using the same encoder as training
    known_labels = set(event_le.classes_)
    unseen = set(data['Extreme Weather Events']) - known_labels
    if unseen:
        print(f"Warning: Unseen labels found in test data: {unseen}. Replacing with 'None'.")
        data['Extreme Weather Events'] = data['Extreme Weather Events'].apply(
            lambda x: x if x in known_labels else 'None'
        )
    data['Extreme Weather Events'] = event_le.transform(data['Extreme Weather Events'].astype(str))

    X = data.drop(['Air Quality Index (AQI)', 'Extreme Weather Events', 'Record ID', 'Date'], axis=1)
    y_aqi = data['Air Quality Index (AQI)']
    y_event = data['Extreme Weather Events']

    # Load models
    aqi_model = joblib.load('aqi_model.pkl')
    event_model = joblib.load('extreme_event_model.pkl')

    # Predict
    aqi_pred = aqi_model.predict(X)
    event_pred = event_model.predict(X)

    # Evaluate
    print("Test MAE (AQI):", mean_absolute_error(y_aqi, aqi_pred))
    print("Test Accuracy (Event):", accuracy_score(y_event, event_pred))
    print("Confusion Matrix (Event):\n", confusion_matrix(y_event, event_pred))
    print("Classification Report (Event):\n", classification_report(
        y_event, event_pred, labels=range(len(event_le.classes_)), target_names=event_le.classes_, zero_division=0))
    print("Predicted Events (labels):", event_le.inverse_transform(event_pred))
    print("Actual Events (labels):", event_le.inverse_transform(y_event))
    print("Predicted AQI:", aqi_pred)
    print("Actual AQI:", list(y_aqi))

    # Show AQI comparison as DataFrame
    result_df = pd.DataFrame({
        "Actual AQI": y_aqi,
        "Predicted AQI": aqi_pred
    })
    print(result_df)

if __name__ == "__main__":
    main()