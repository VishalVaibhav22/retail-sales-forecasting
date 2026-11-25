#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# In[2]:


URL = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv"

df = pd.read_csv(URL, parse_dates=["ds"])

df = df.rename(columns={
    "ds": "date",
    "y": "sales"
})

df = df.sort_values("date").reset_index(drop=True)

print(df.head())


# In[3]:


print("Number of rows:", len(df))
print("Starting date:", df["date"].min())
print("Ending date:", df["date"].max())

print("\nMissing values:")
print(df.isna().sum())


# In[4]:


expected_dates = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="MS"
)

missing_dates = expected_dates.difference(df["date"])

print("Expected number of months:", len(expected_dates))
print("Missing months:", len(missing_dates))
print(missing_dates)


# In[5]:


plt.figure(figsize=(12, 5))

plt.plot(df["date"], df["sales"])

plt.title("Monthly US Retail Sales (1992–2016)")
plt.xlabel("Date")
plt.ylabel("Retail Sales")

plt.grid(True)
plt.tight_layout()

plt.savefig("sales_history.png", dpi=150)

plt.show()


# In[6]:


TEST_MONTHS = 36
SEED = 42

n = len(df)

y = df["sales"].to_numpy(dtype=float)

test_idx = list(range(n - TEST_MONTHS, n))

print("First test month:", df.loc[test_idx[0], "date"])
print("Last test month:", df.loc[test_idx[-1], "date"])
print("Number of test months:", len(test_idx))


# In[7]:


def mape(actual, predicted):
    return float(
        np.mean(np.abs((actual - predicted) / actual)) * 100
    )


def rmse(actual, predicted):
    return float(
        np.sqrt(np.mean((actual - predicted) ** 2))
    )


# In[8]:


actual = []
p_naive = []
p_snaive = []

for t in test_idx:

    actual.append(y[t])

    # Predict using the previous month
    p_naive.append(y[t - 1])

    # Predict using the same month last year
    p_snaive.append(y[t - 12])


actual = np.array(actual)

p_naive = np.array(p_naive)

p_snaive = np.array(p_snaive)


print("Naive MAPE:", round(mape(actual, p_naive), 2))

print("Seasonal Naive MAPE:", round(mape(actual, p_snaive), 2))


# In[9]:


print("Actual observations:", len(actual))
print("Naive predictions:", len(p_naive))
print("Seasonal predictions:", len(p_snaive))


# In[10]:


p_sarima = []

for t in test_idx:

    model = SARIMAX(
        y[:t],
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)

    forecast = fitted_model.forecast(steps=1)

    p_sarima.append(float(np.asarray(forecast)[0]))


p_sarima = np.array(p_sarima)

print("SARIMA MAPE:", round(mape(actual, p_sarima), 2))
print("SARIMA RMSE:", round(rmse(actual, p_sarima), 0))


# In[13]:


logy = np.log(y)

first_t = 25

print("Original sales:", y[:5])
print("Log sales:", logy[:5])


# In[14]:


def gbm_features(series_log, t):

    def growth(k):
        return series_log[t-k] - series_log[t-k-12]

    return [
        growth(1),
        growth(2),
        growth(3),
        growth(12) if t-24 >= 0 else 0.0,
        series_log[t-1] - series_log[t-2],
        t % 12
    ]


# In[15]:


sample_features = gbm_features(logy, 25)

print("Features:", sample_features)
print("Number of features:", len(sample_features))


# In[16]:


GradientBoostingRegressor(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    random_state=SEED
)


# In[17]:


p_gbm = []

for t in test_idx:

    # Step 1: Create training features
    X = np.array([
        gbm_features(logy, s)
        for s in range(first_t, t)
    ])

    # Step 2: Create training targets
    target = np.array([
        logy[s] - logy[s - 12]
        for s in range(first_t, t)
    ])

    # Step 3: Train Gradient Boosting
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        random_state=SEED
    ).fit(X, target)

    # Step 4: Predict growth
    growth = gb.predict(
        np.array([gbm_features(logy, t)])
    )[0]

    # Step 5: Convert growth into sales
    prediction = y[t - 12] * np.exp(growth)

    p_gbm.append(float(prediction))


p_gbm = np.array(p_gbm)


# In[18]:


print("Gradient Boosting MAPE:", round(mape(actual, p_gbm), 2))

print("Gradient Boosting RMSE:", round(rmse(actual, p_gbm), 0))


# In[19]:


results = {
    "Naive (last month)": p_naive,

    "Seasonal naive (same month last year)": p_snaive,

    "SARIMA(1,1,1)(0,1,1,12)": p_sarima,

    "Gradient boosting (YoY growth)": p_gbm
}


# In[20]:


rows = []

for model_name, predictions in results.items():

    rows.append({
        "model": model_name,
        "MAPE_%": round(mape(actual, predictions), 2),
        "RMSE": round(rmse(actual, predictions), 0)
    })


table = pd.DataFrame(rows)

table = table.sort_values("MAPE_%")

print(table.to_string(index=False))


# In[21]:


out = pd.DataFrame({
    "date": df.loc[test_idx, "date"].values,
    "actual": actual
})

for model_name, predictions in results.items():

    out[model_name] = predictions


out.to_csv("forecast_vs_actual.csv", index=False)

print("Forecast results exported!")


# In[22]:


table.to_csv("model_metrics.csv", index=False)

print("Model metrics exported!")


# In[23]:


plt.figure(figsize=(14, 6))

plt.plot(
    df.loc[test_idx, "date"],
    actual,
    label="Actual",
    linewidth=2
)

for model_name, predictions in results.items():

    plt.plot(
        df.loc[test_idx, "date"],
        predictions,
        label=model_name
    )

plt.title("Actual vs Forecasted US Retail Sales")
plt.xlabel("Date")
plt.ylabel("Retail Sales")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("forecast_comparison.png", dpi=150)

plt.show()


# In[ ]:




