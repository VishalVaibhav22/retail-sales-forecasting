

\# US Retail Sales Forecasting



\## Overview



A time-series forecasting project that predicts monthly US retail sales using four approaches:



\- Naive Forecasting

\- Seasonal Naive Forecasting

\- SARIMA

\- Gradient Boosting Regression



The objective is to compare statistical and machine learning forecasting methods using walk-forward validation.



\## Dataset



\- Source: \[Facebook Prophet GitHub](https://raw.githubusercontent.com/facebook/prophet/main/examples/example\_retail\_sales.csv)

\- Frequency: Monthly

\- Period: January 1992 – May 2016

\- Observations: 293



\## Methodology



Used one-month-ahead walk-forward validation over the final 36 months (June 2013 – May 2016).



Each model was trained only on historical observations available before the month being forecast.



\### Models



1\. Naive: Uses the previous month's sales.

2\. Seasonal Naive: Uses sales from the same month in the previous year.

3\. SARIMA: SARIMA(1,1,1)(0,1,1,12).

4\. Gradient Boosting: Predicts year-over-year log growth using engineered historical features.



\## Model Performance



| Model | MAPE (%) | RMSE |

|---|---:|---:|

| SARIMA | 1.30 | 6,944 |

| Gradient Boosting | 1.44 | 7,575 |

| Seasonal Naive | 3.23 | 15,522 |

| Naive | 6.70 | 41,451 |



\## Visualizations



\### Historical Retail Sales



!\[Historical Sales](sales\_history.png)



\### Forecast Comparison



!\[Forecast Comparison](forecast\_comparison.png)



\## Technologies Used



Python, Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib.



\## Limitations



\- Uses one aggregate national sales series rather than store-level or product-category data.

\- Evaluation covers only one 36-month test window.

\- Model parameters were selected once and not tuned.

\- Only one-month-ahead forecasting was evaluated.

\- Results may not generalize to other retailers or markets.



\## Future Improvements



\- Incorporate promotional and pricing information.

\- Evaluate longer forecasting horizons.

\- Test additional forecasting models.

\- Extend the analysis to individual retail categories.



