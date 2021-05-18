# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data analysis: Covid cases over time

# %% [markdown]
# Data analysis of daily new covid cases in different countries, using data from:
#
# https://github.com/CSSEGISandData/COVID-19

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## Import and clean up data

# %%
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# %%
df.head(3)

# %% [markdown]
# ### Remove unused columns

# %%
df = df.drop(["Lat", "Long"], axis=1)

# %% tags=[]
df.head(3)

# %% [markdown]
# ### Convert from wide format to long

# %%
df = df.melt(id_vars=["Province/State", "Country/Region"], var_name="Date", value_name="Cases")

# %%
df.head(3)

# %% [markdown]
# ### Make values in date column proper dates

# %%
df.dtypes

# %%
df["Date"] = pd.to_datetime(df["Date"])

# %%
df.dtypes

# %%
df.head(3)

# %% [markdown]
# ### Sum provinces, aggregate to country level

# %% tags=[]
df[(df["Country/Region"] == "United Kingdom") & 
   (df["Date"] == pd.to_datetime("2021-01-01"))]

# %%
df = df.groupby(["Country/Region", "Date"], as_index=False).sum()

# %%
df[(df["Country/Region"] == "United Kingdom") &
   (df["Date"] == pd.to_datetime("2021-01-01"))]

# %% [markdown]
# ### Convert cumulative total to daily new cases

# %%
df[(df["Country/Region"] == "Poland") &
   (df["Date"] >= pd.to_datetime("2021-01-01")) &
   (df["Date"] <= pd.to_datetime("2021-01-03"))]

# %%
df["Cases"] = df.groupby("Country/Region")["Cases"].transform("diff")

# %%
df[(df["Country/Region"] == "Poland") &
   (df["Date"] >= pd.to_datetime("2021-01-01")) &
   (df["Date"] <= pd.to_datetime("2021-01-03"))]

# %% [markdown]
# ## Analyze

# %%
date_from = pd.to_datetime("2021-01-01")
date_to   = pd.to_datetime("2021-04-22")

countries = sorted([
    "Poland",
    "Czechia",
    "Germany",
    "Austria"
])


# %% [markdown]
# ### Day by day plot

# %%
def cases_day_by_day():
    return df[df["Country/Region"].isin(countries) &
              (df["Date"] >= date_from) &
              (df["Date"] <= date_to)]

sns.relplot(data=cases_day_by_day(), x="Date", y="Cases", hue="Country/Region", kind="line", aspect=3)


# %% [markdown]
# ### Week by week plot

# %%
def cases_week_by_week():
    def week_start(dt):
        return dt - timedelta(days=dt.weekday())

    return (
        df.copy()
        .loc[df["Country/Region"].isin(countries) &
             (df["Date"] >= week_start(date_from)) &
             (df["Date"] < week_start(date_to))]
        .groupby("Country/Region")
        .resample("1W", on="Date")
        .sum()
    )

sns.relplot(data=cases_week_by_week(), x="Date", y="Cases", hue="Country/Region", kind="line", marker="o", aspect=3)


# %% [markdown]
# ### 7 day moving average plot

# %%
def cases_moving_average(days):
    # To compute the moving average you need more days than will be presented
    ma_date_from = date_from - timedelta(days=days)

    return (
        df.copy()
        .loc[df["Country/Region"].isin(countries) &
             (df["Date"] >= ma_date_from) &
             (df["Date"] <= date_to)]
        .groupby("Country/Region")
        .rolling(days, on="Date")
        .mean()
    )

sns.relplot(data=cases_moving_average(7), x="Date", y="Cases", hue="Country/Region", kind="line", aspect=3)
