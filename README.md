# Climate Change Analysis: India AQI & Extreme Weather Prediction

This project uses machine learning to predict Air Quality Index (AQI) and Extreme Weather Events for Indian cities based on climate data. The predictions are exported for visualization in Power BI.

---

## Project Structure

```
requirements.txt
data/
    india_climate_data.csv
src/
    train_aqi_model.py
    train_extreme_event_model.py
    export_for_powerbi.py
aqi_model.pkl
extreme_event_model.pkl
predictions_for_powerbi.csv
```

---

## Workflow

1. **Install requirements**
   ```sh
   pip install -r requirements.txt
   ```

2. **Train the AQI prediction model**
   ```sh
   python src/train_aqi_model.py
   ```

3. **Train the Extreme Weather Event prediction model**
   ```sh
   python src/train_extreme_event_model.py
   ```

4. **Export predictions for Power BI**
   ```sh
   python src/export_for_powerbi.py
   ```

   This will generate `predictions_for_powerbi.csv` for use in Power BI.

---

## File Descriptions

- **requirements.txt**: Python dependencies.
- **data/india_climate_data.csv**: Input climate dataset.
- **src/train_aqi_model.py**: Trains and saves the AQI regression model.
- **src/train_extreme_event_model.py**: Trains and saves the extreme weather classification model.
- **src/export_for_powerbi.py**: Generates predictions using both models and exports results for Power BI.
- **aqi_model.pkl**, **extreme_event_model.pkl**: Saved trained models.
- **predictions_for_powerbi.csv**: Output file for Power BI.

---

## Power BI Integration

1. Open Power BI Desktop.
2. Click **Get Data** > **Text/CSV**.
3. Select `predictions_for_powerbi.csv` and load the data.
4. Create your visualizations using the predicted AQI and Extreme Weather Event columns.

---

## Notes

- Ensure your `india_climate_data.csv` file matches the expected column names (e.g., `Air Quality Index (AQI)`, `Extreme Weather Events`).
- All scripts should be run from the project root directory.

---
