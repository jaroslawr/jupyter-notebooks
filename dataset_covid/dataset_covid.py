# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
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
# ### Make the date a column

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
# ## Keep only data starting from 2021

# %%
df.head(5)

# %%
df = df[df["Date"] >= pd.to_datetime("2021-01-01")]

# %%
df.head(5)

# %% [markdown]
# ### Sum provinces, aggregate to country level

# %% tags=[]
df[(df["Country/Region"] == "United Kingdom") & 
   (df["Date"] == pd.to_datetime("2021-01-01"))]

# %%
df = df.groupby(["Country/Region", "Date"]).sum()

# %%
df.loc["United Kingdom", pd.to_datetime("2021-01-01")]

# %% [markdown]
# ### Convert cumulative total to daily new cases

# %%
df.loc["Poland", :].iloc[0:5]

# %%
df = df.groupby(level=0).diff()

# %%
df.loc["Poland", :].iloc[0:5]

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
# ### Plot cases day by day

# %%
def plot_cases_day_by_day():
    (df.loc[(countries, slice(date_from, date_to)), :]
     .unstack(level=0)
     .loc[:, "Cases"]
     .plot(figsize=(16,5)))


# %%
plot_cases_day_by_day()


# %% [markdown]
# ### Plot cases week-by-week

# %%
def plot_cases_week_by_week():
    def week_start(dt):
        return dt - timedelta(days=dt.weekday())

    (df.loc[(countries, slice(date_from, date_to)), :]
     .groupby(level=0)
     .resample("1W", level=1)
     .sum()
     .unstack(level=0)
     .loc[:, "Cases"]
     .plot(figsize=(16, 5)))


# %%
plot_cases_week_by_week()


# %% [markdown]
# ### Plot moving average of cases

# %%
def plot_moving_average_of_cases(days):
    # To compute the moving average you need more days than will be presented
    ma_date_from = date_from - timedelta(days=days)

    (df.loc[(countries, slice(ma_date_from, date_to)), :]
     .groupby(level=0)
     # rolling() does not support multi-indexes or level argument at the moment,
     # the apply() call works around this
     .apply(lambda df: df.rolling(days).sum())
     .loc[:, "Cases"]
     .unstack(level=0)
     .plot(figsize=(16,5)))


# %%
plot_moving_average_of_cases(days=7)
