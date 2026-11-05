# Model Card — Kolkata Next-Day AQI Predictor

## Model Details

- **Task**: Next-day Air Quality Index (AQI) regression for Kolkata, India.
- **Input**: Daily lag/rolling features of PM2.5, PM10, NO₂, SO₂, O₃, CO.
- **Output**: Predicted AQI for the following day.
- **Architecture**: Tabular regression (RandomForest / GradientBoosting / Ridge / SVR).
- **Data source**: OpenAQ API v2 (free, open-source).

## Intended Use

- Research and educational forecasting of Kolkata air quality.
- Public-awareness dashboards (Streamlit app).
- NOT a substitute for official government AQI bulletins or health advice.

## Training Details

- **Loss**: Implicit squared error (regression).
- **Optimizer**: scikit-learn defaults (tree ensembles: least-squares split; SVR: SMO).
- **Compute**: CPU only; 16 GB RAM, no GPU required.
- **Time to train**: seconds to a few minutes depending on data size.

## Evaluation

| Model | Test MAE | Test RMSE | Test R² |
|-------|----------|-----------|---------|
| *(run `python -m src.scripts.run_pipeline` to populate)* | | | |

## Limitations

- Predictions are estimates; actual AQI can vary due to unmodelled factors
  (weather, traffic, Diwali fireworks, inversion layers).
- Data gaps in OpenAQ observations may bias daily aggregates.
- Model is specific to Kolkata and should not be transferred without
  retraining.

## Ethical Considerations

AQI predictions can influence public behaviour and health decisions. This
model is provided as-is for research; users must cross-check with official
sources before making health-related decisions.