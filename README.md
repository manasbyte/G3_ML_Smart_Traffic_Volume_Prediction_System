# TrafficIQ — Smart Traffic Volume Prediction System

A working educational PBL prototype with a Streamlit multi-page GUI, synthetic traffic dataset, Random Forest regression model, Excel database, road comparison and traffic relief suggestions.

## Included
- `app.py` — login and 4 application pages
- `generate_dataset.py` — creates synthetic traffic data in CSV and Excel
- `train_model.py` — trains and evaluates the Random Forest model
- `traffic_dataset.csv` — generated dataset
- `traffic_database.xlsx` — Excel database with Traffic_Data, Road_Hour_Summary and Data_Dictionary sheets
- `traffic_model.pkl` — trained model
- `model_metrics.json` — evaluation metrics
- `prediction_history.csv` — created after predictions are run
- `requirements.txt` — Python dependencies

## Run on macOS / Windows
Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

On Windows Command Prompt, activate with:
```bat
.venv\Scripts\activate
```
Then use `python` instead of `python3` if required.

The app opens in your browser, usually at `http://localhost:8501`.

## Demo login
Username: `admin`
Password: `traffic123`

This is intentionally a demo login only: any non-empty username and password are accepted. Do not use real credentials.

## Dataset and scientific limitations
The included dataset is synthetic and generated from programmed assumptions about time of day, road, weather, weekends and holidays. It is useful for demonstrating an end-to-end ML workflow, but it is not real sensor data. Metrics describe performance on a held-out subset of synthetic data and must not be presented as verified real-world accuracy. The road advisor compares predicted volume only; it does not calculate actual distance, travel time or live navigation.

## Upload guidance
- Experiment Code: upload a ZIP of this folder or a PDF/DOCX containing code, depending on your instructor's accepted format.
- Dataset: upload `traffic_database.xlsx` or `traffic_dataset.csv` to Google Drive and share a viewable link; the form asks for link only.
- PPT and Report: create after testing the app and taking screenshots so reported results match the working model.
