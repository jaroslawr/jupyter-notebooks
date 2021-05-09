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
# ## Imports

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timedelta

# %% [markdown]
# ## Setup

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## Read data

# %%
df = pd.read_csv("time_series_covid19_confirmed_global.csv")

# %%
df.head(3)

# %% [markdown]
# ## Remove unused columns

# %%
df = df.drop(["Lat", "Long"], axis=1)

# %% tags=[]
df.head(3)

# %% [markdown]
# ## Convert from wide format to long

# %%
df = df.melt(id_vars=["Province/State", "Country/Region"], var_name="Date", value_name="Cases")

# %%
df.head(3)

# %% [markdown]
# ## Make values in date column proper dates

# %%
df.dtypes

# %%
df["Date"] = pd.to_datetime(df["Date"])

# %%
df.dtypes

# %%
df.head(3)

# %% [markdown]
# ## Sum provinces, aggregate to country level

# %% tags=[]
df[(df["Country/Region"] == "United Kingdom") & 
   (df["Date"] == datetime(2021, 1, 1))]

# %%
df = df.groupby(["Country/Region", "Date"], as_index=False).sum()

# %%
df[(df["Country/Region"] == "United Kingdom") &
   (df["Date"] == datetime(2021, 1, 1))]

# %% [markdown]
# ## Convert cumulative total to daily new cases

# %%
df[(df["Country/Region"] == "Poland") &
   df["Date"].isin(pd.date_range(datetime(2021, 1, 1), datetime(2021, 1, 3)))]

# %%
df["Cases"] = df.groupby("Country/Region")["Cases"].transform("diff")

# %%
df[(df["Country/Region"] == "Poland") &
   df["Date"].isin(pd.date_range(datetime(2021, 1, 1), datetime(2021, 1, 3)))]

# %% [markdown]
# ## Analyze the data

# %%
date_from = datetime(2021, 1, 1)
date_to   = datetime(2021, 4, 22)

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
    date_range = pd.date_range(date_from, date_to)
    return df[df["Country/Region"].isin(countries) &
              df["Date"].isin(date_range)]

sns.relplot(data=cases_day_by_day(), x="Date", y="Cases", hue="Country/Region", kind="line", aspect=3)


# %% [markdown]
# ### Week by week plot

# %%
def cases_week_by_week():
    def week_start(dt):
        return dt - timedelta(days=dt.weekday())

    date_range = pd.date_range(week_start(date_from), week_start(date_to), closed="left")
    return (
        df.copy()
        .loc[df["Country/Region"].isin(countries) &
             df["Date"].isin(date_range)]
        .groupby("Country/Region")
        .resample("1W", on="Date")
        .sum()
    )

sns.relplot(data=cases_week_by_week(), x="Date", y="Cases", hue="Country/Region", kind="line", marker="o", aspect=3)


# %% [markdown]
# ### 7 day moving average plot

# %%
def cases_moving_average(days):
    plot_date_range = pd.date_range(date_from, date_to)
    # You need more days than will be presented to compute the moving average
    moving_window_date_range = pd.date_range(date_from - timedelta(days=days), date_to)

    return (
        df.copy()
        .loc[df["Country/Region"].isin(countries) &
             df["Date"].isin(moving_window_date_range)]
        .groupby("Country/Region")
        .rolling(days, on="Date")
        .mean()
    )

sns.relplot(data=cases_moving_average(7), x="Date", y="Cases", hue="Country/Region", kind="line", aspect=3)
