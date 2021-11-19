# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
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

from datetime import datetime, timedelta

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## Import and clean up data

# %% [markdown]
# ### Read the CSV

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
df["Date"] = pd.to_datetime(df["Date"])

# %%
df.dtypes

# %%
df.head(3)

# %% [markdown] tags=[]
# ### Sum provinces, aggregate to country level

# %% tags=[]
df[(df["Country/Region"] == "United Kingdom") & (df["Date"] == df["Date"].max())]

# %%
df = df.groupby(["Country/Region", "Date"]).sum()

# %% [markdown]
# Dataframe is now indexed by country and date and selection can be done via `loc[]`:

# %%
df.loc["United Kingdom"].iloc[-1]

# %% [markdown] tags=[]
# ### Convert cumulative total to daily new cases

# %%
df.loc["United Kingdom"].iloc[-5:]

# %%
df = df.groupby(level=0).diff()

# %%
df.loc["United Kingdom"].iloc[-5:]

# %% [markdown] tags=[]
# ### Keep only data starting from 2020-06

# %%
df.head(3)

# %%
df = df.loc[(slice(None), slice(pd.to_datetime("2020-06-01"), None)), :]

# %% tags=[]
df.head(3)

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


# %%
def plot(df):
    fig, ax = plt.subplots(figsize=(16,5))
    df.unstack(level="Country/Region").plot(ax=ax, legend=False)
    fig.legend(loc="center right")


# %% [markdown]
# ### Plot cases day by day

# %%
cases_day_by_day_df = df.loc[(countries, slice(date_from, date_to)), "Cases"]
cases_day_by_day_df.groupby(level="Country/Region").head()

# %%
plot(cases_day_by_day_df)


# %% [markdown]
# ### Plot cases week-by-week

# %%
def week_start(dt):
    return dt - timedelta(days=dt.weekday())


# %%
cases_week_by_week_df = (
    df.loc[(countries, slice(week_start(date_from), date_to)), "Cases"]
    .groupby(level="Country/Region")
    .resample("1W", level="Date")
    .sum()
)
cases_week_by_week_df.groupby(level="Country/Region").head()

# %%
plot(cases_week_by_week_df)

# %% [markdown]
# ### Plot moving average of cases

# %%
cases_moving_average_df = (
    df.loc[(countries, slice(date_from - timedelta(days=7), date_to)), "Cases"]
    .groupby(level="Country/Region")
    .apply(lambda df: df.rolling(7).sum())
    .loc[(countries, slice(date_from, date_to))]
)
cases_moving_average_df.groupby(level="Country/Region").head(10)

# %%
plot(cases_moving_average_df)
