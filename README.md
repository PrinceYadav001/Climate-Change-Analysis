# Climate Change Analysis: India

This repository contains a comprehensive machine learning pipeline for analyzing and predicting climate change impacts in India, focusing on Air Quality Index (AQI) and Extreme Weather Events. The project is designed for data science analysis and seamless Power BI visualization.

---

## 🌏 Project Motivation & Objective

Climate change is leading to more frequent extreme weather events and deteriorating air quality, especially in developing countries like India.

**Objectives:**
- Predict AQI and extreme weather events for Indian cities using machine learning.
- Provide actionable insights and visualizations for policymakers, researchers, and the public.

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
├── README.md                            # Project documentation
│
PowerBI Dashboards/
│   ├── Dashboard - 1.pbix               # Power BI Dashboard 1
│   └── Dashboard - 2.pbix               # Power BI Dashboard 2
```

---

## ✅ Key Steps Covered

| Step                                               | Status | Where & How Implemented                                                  |
|----------------------------------------------------|--------|--------------------------------------------------------------------------|
| Identifying and sourcing relevant datasets         | ✔️     | Used multi-featured, real-world climate datasets in `/data`              |
| Cleaning and handling missing values               | ✔️     | Used `data.ffill()` and `.fillna('None')` for missing values             |
| Feature selection and engineering                  | ✔️     | Selected relevant features, encoded categoricals, dropped irrelevant cols |
| Ensuring data integrity and consistency            | ✔️     | Consistent encoders/models, robust label handling in all scripts         |
| Summary statistics and insights                    | ✔️     | Printed MAE, accuracy, confusion matrix, feature importances             |
| Identifying patterns, trends, and anomalies        | ✔️     | Used model evaluation, feature importance, Power BI-ready output         |
| Handling outliers and data transformations         | ✔️     | Used robust models, consistent data transformations                      |
| Initial visual representation of key findings      | ✔️     | Exported CSV for Power BI, DataFrame prints, sample tables, dashboards   |

---

## 🚀 Workflow

### 1. **Train Models**
- `src/train_aqi_model.py`: Trains a RandomForestRegressor to predict AQI using climate and city features.
- `src/train_extreme_event_model.py`: Trains a classifier to predict extreme weather events.

### 2. **Test Models**
- `src/predict_on_test.py`: Evaluates both models on test data, prints metrics (MAE, accuracy, confusion matrix, classification report), and compares actual vs predicted values.

### 3. **Export for Power BI**
- `src/export_for_powerbi.py`: Generates `predictions_for_powerbi.csv` with actual and predicted AQI and events for visualization.

### 4. **Power BI Dashboards**
- `PowerBI Dashboards/Dashboard - 1.pbix` and `Dashboard - 2.pbix`: Interactive dashboards for visualizing predictions and insights.

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

5. **Open Power BI Dashboards**
    - Open `.pbix` files in the `PowerBI Dashboards` folder using Power BI Desktop.

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

- **Power BI Dashboards**  
  - Visualize trends, patterns, and predictions interactively.

---

## 📈 Summary Statistics & Insights

- **AQI Model:**  
  - Mean Absolute Error (Train): ~124.5  
  - Mean Absolute Error (Test): ~77.1  
  - **Feature Importances:** Economic Impact, Population Exposure, Humidity, Precipitation, Wind Speed

- **Extreme Event Model:**  
  - Accuracy (Train): ~27%  
  - Accuracy (Test): ~73%  
  - **Confusion Matrix & Classification Report:** Model predicts common events well, struggles with rare events due to class imbalance.

---

## 📉 Patterns, Trends, and Anomalies

- **Patterns:**  
  - AQI is influenced most by economic impact, population exposure, and weather features.
  - Most test samples are "None" or "Flood" for events.

- **Trends:**  
  - Model generalizes well for AQI and frequent events.
  - Class imbalance affects rare event prediction.

- **Anomalies:**  
  - Some rare events (Drought, Hurricane) are underrepresented and harder to predict.

---

## 📊 Initial Visual Representation

- **Power BI Ready:**  
  - Exported `predictions_for_powerbi.csv` is structured for easy import into Power BI.
  - Columns: Date, City, Actual AQI, Actual Event, Predicted AQI, Predicted Event

- **Sample Table:**

| Date      | City | Air Quality Index (AQI) | Actual_Event | Predicted_AQI | Predicted_Event |
|-----------|------|-------------------------|--------------|---------------|-----------------|
| 1/1/2024  | 1    | 276                     | None         | 269.97        | None            |
| 1/3/2024  | 1    | 472                     | None         | 350.90        | Flood           |
| 1/14/2024 | 1    | 453                     | Flood        | 361.90        | Flood           |
| 1/15/2024 | 1    | 82                      | Flood        | 161.58        | Heatwave        |

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

## 📄 License

This project is for educational and research purposes.

---

## 👤 Author

[By Team ShadowHart](https://github.com/PrinceYadav001)

---

## 💡 Improvements

- Try advanced ML models or hyperparameter tuning for better accuracy.
- Address class imbalance for rare extreme events.
- Add more features or external data for richer analysis.
- Enhance Power BI dashboards with more visual insights.