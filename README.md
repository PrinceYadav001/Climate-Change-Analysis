# Climate Change Analysis: India

This repository contains a complete machine learning pipeline for analyzing and predicting climate change impacts in India, focusing on Air Quality Index (AQI) and Extreme Weather Events. The project is designed for data science analysis and Power BI visualization.

---

## 📁 Project Structure

```
india-climate-ml-pbi/
│
├── data/
│   ├── india_climate_data.csv           # Main training data
│   └── india_climate_test_data.csv      # Test data for evaluation
│
├── src/
│   ├── train_aqi_model.py               # Train AQI regression model
│   ├── train_extreme_event_model.py     # Train extreme event classifier
│   ├── predict_on_test.py               # Evaluate models on test data
│   └── export_for_powerbi.py            # Export predictions for Power BI
│
├── aqi_model.pkl                        # Saved AQI model
├── extreme_event_model.pkl              # Saved event classifier
├── aqi_label_encoders.pkl               # Saved encoders for categorical features
├── event_label_encoder.pkl              # Saved encoder for event labels
├── predictions_for_powerbi.csv          # Output for Power BI
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## 🚀 Workflow

### 1. **Train Models**
- `src/train_aqi_model.py`: Trains a RandomForestRegressor to predict AQI using climate and city features.
- `src/train_extreme_event_model.py`: Trains a classifier to predict extreme weather events.

### 2. **Test Models**
- `src/predict_on_test.py`: Evaluates both models on test data, prints metrics (MAE, accuracy, confusion matrix, classification report), and compares actual vs predicted values.

### 3. **Export for Power BI**
- `src/export_for_powerbi.py`: Generates `predictions_for_powerbi.csv` with actual and predicted AQI and events for visualization.

---

## 📊 Data

- **`india_climate_data.csv`**:  
  Contains columns such as:
  - Record ID, Date, Country, City, Temperature (°C), Humidity (%), Precipitation (mm), Air Quality Index (AQI), Extreme Weather Events, Climate Classification, Climate_Zone, Biome_Type, Heat_Index, Wind_Speed, Wind_Direction, Season, Population_Exposure, Economic_Impact_Estimate, Infrastructure_Vulnerability_Score

- **`india_climate_test_data.csv`**:  
  Same structure as above, used for model evaluation.

---

## 🛠️ How to Run

1. **Install requirements**
    ```sh
    pip install -r india-climate-ml-pbi/requirements.txt
    ```

2. **Train models**
    ```sh
    python india-climate-ml-pbi/src/train_aqi_model.py
    python india-climate-ml-pbi/src/train_extreme_event_model.py
    ```

3. **Test models**
    ```sh
    python india-climate-ml-pbi/src/predict_on_test.py
    ```

4. **Export for Power BI**
    ```sh
    python india-climate-ml-pbi/src/export_for_powerbi.py
    ```

---

## 📦 Output

- **`predictions_for_powerbi.csv`**  
  Contains columns:
  - Date
  - City
  - Air Quality Index (AQI) (actual)
  - Actual_Event (actual extreme event)
  - Predicted_AQI
  - Predicted_Event

  This file is ready for import into Power BI for visualization and further analysis.

---

## ⚠️ Notes & Best Practices

- **File Paths:**  
  Always use forward slashes (`/`) in file paths for cross-platform compatibility.

- **Missing Values:**  
  All missing values in "Extreme Weather Events" are handled as `"None"` in both training and prediction.

- **Encoders:**  
  All encoders are saved as `.pkl` files for reuse and consistency between training and prediction.

- **Class Imbalance:**  
  The dataset may be imbalanced for rare events. Consider using resampling or class weights for better event prediction.

---

## 📊 Example Output

A sample from `predictions_for_powerbi.csv`:

| Date      | City | Air Quality Index (AQI) | Actual_Event | Predicted_AQI | Predicted_Event |
|-----------|------|-------------------------|--------------|---------------|-----------------|
| 1/1/2024  | 1    | 276                     | None         | 269.97        | None            |
| 1/3/2024  | 1    | 472                     | None         | 350.90        | Flood           |
| 1/14/2024 | 1    | 453                     | Flood        | 361.90        | Flood           |
| 1/15/2024 | 1    | 82                      | Flood        | 161.58        | Heatwave        |

---

## 📄 License

This project is for educational and research purposes.

---

## 👤 Author

[PrinceYadav001](https://github.com/PrinceYadav001)

---

## 💡 Improvements

- Try advanced ML models or hyperparameter tuning for better accuracy.
- Address class imbalance for rare extreme events.
- Add more features or external data for richer analysis.
