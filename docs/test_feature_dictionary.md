# Feature Dictionary — AQI Forecasting

## Numeric Features
- aqi
- pm2.5
- pm10
- aqi_lag1
- aqi_lag3
- aqi_lag7
- aqi_lag14
- aqi_roll3_mean
- aqi_roll7_mean
- aqi_roll7_std
- aqi_next_day
- pm2.5_lag1
- pm10_lag1

## Categorical Features
- season

## Target Variable
- **Name:** aqi_next_day
- **Description:** Air Quality Index (AQI) for the next calendar day.

## Lag Features
- aqi_lag1

- aqi_lag3

- aqi_lag7

- aqi_lag14

- pm2.5_lag1

- pm10_lag1

## Rolling Statistics Features
- aqi_roll3_mean


- aqi_roll7_mean


- aqi_roll7_std


## Temporal Features
- day_of_week

- month

- season

