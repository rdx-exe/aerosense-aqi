# Model Card — AeroSense Kolkata Next-Day AQI Predictor

## Model Details

- **Task**: Next-day Air Quality Index (AQI) regression for Kolkata, India.
- **Input**: Daily lag/rolling features of PM2.5, PM10, NO₂, SO₂, O₃, CO.
- **Output**: Predicted AQI for the following day.
- **Architecture**: Ridge regression (`alpha=10`) on the 20 features selected by
  random-forest importance. Random Forest, Extra Trees, Gradient Boosting, and SVR
  are also compared on the validation split.
- **Data source**: Open-Meteo Air Quality API (hourly CAMS/ECMWF fields,
  daily means) for 2024-01-01 → 2026-11-06. Modelled values, not a live official
  CPCB station feed.

## Intended Use

- Research and educational forecasting of Kolkata air quality.
- Public-awareness dashboards (Streamlit app).
- NOT a substitute for official government AQI bulletins or health advice.

## Training Details

- **Training window**: 2024-01-08 → 2025-12-29 (722 rows).
- **Validation window**: 2025-12-31 → 2026-06-01 (153 rows), used for model selection.
- **Test window**: 2026-06-03 → 2026-11-05 (156 rows), never used for training or
  selection.
- **Loss**: Squared error (Ridge).
- **Hyperparameters**: Selected with expanding-window `TimeSeriesSplit`
  cross-validation on the training split; validation RMSE used for the final
  candidate comparison.
- **Compute**: CPU only; seconds to a few minutes.

## Evaluation

Final independent test evaluation (2026-06-03 to 2026-11-05, 156 daily rows):

| Model | Test MAE | Test RMSE | Test R² |
|-------|----------|-----------|---------|
| Ridge regression (selected on validation) | 16.70 | 21.08 | 0.773 |
| Persistence baseline | 15.96 | 21.97 | 0.753 |

The model improves RMSE and R² over persistence but is slightly worse on MAE.
It remains compressed toward the training mean and under-predicts pollution peaks.
See [`documentation/model_evaluation.md`](../documentation/model_evaluation.md)
for the full error analysis and integrity statement.

## Limitations

- Predictions are estimates; actual AQI can vary due to unmodelled factors
  (weather, traffic, Diwali fireworks, inversion layers).
- The input series is a modelled/reanalysis product, not direct monitor data.
- Model is specific to Kolkata and should not be transferred without
  retraining.
- Predictions are compressed toward the training mean; the model is not
  reliable for extreme-event alerts.

## Ethical Considerations

AQI predictions can influence public behaviour and health decisions. This
model is provided as-is for research; users must cross-check with official
sources before making health-related decisions.
