# Model Card — Kolkata Next-Day AQI Predictor

## Model Details

- **Task**: Next-day Air Quality Index (AQI) regression for Kolkata, India.
- **Input**: Daily lag/rolling features of PM2.5, PM10, NO₂, SO₂, O₃, CO.
- **Output**: Predicted AQI for the following day.
- **Architecture**: Tabular regression (RandomForest selected on validation; GradientBoosting, Ridge and SVR also compared).
- **Data source**: Repository's processed 2-year prototyping dataset (CPCB-style pollutant series); not a live official feed.

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

Final independent test evaluation (2023-09-13 to 2023-12-30, 109 daily rows):

| Model | Test MAE | Test RMSE | Test R² |
|-------|----------|-----------|---------|
| Random Forest (selected on validation) | 18.76 | 25.30 | 0.011 |
| Persistence baseline | 23.76 | 32.06 | -0.588 |

The model reduces average error versus persistence but has near-zero R² and
systematically under-predicts extreme pollution days. See
[`documentation/model_evaluation.md`](../documentation/model_evaluation.md) for
the full error analysis and integrity statement.

## Limitations

- Predictions are estimates; actual AQI can vary due to unmodelled factors
  (weather, traffic, Diwali fireworks, inversion layers).
- Data gaps in the source observations may bias daily aggregates.
- Model is specific to Kolkata and should not be transferred without
  retraining.
- Predictions are compressed toward the training mean; the model is not
  reliable for extreme-event alerts.

## Ethical Considerations

AQI predictions can influence public behaviour and health decisions. This
model is provided as-is for research; users must cross-check with official
sources before making health-related decisions.