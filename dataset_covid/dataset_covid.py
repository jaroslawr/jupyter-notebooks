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
df = df.groupby(level="Country/Region").diff()

# %%
df.loc["United Kingdom"].iloc[-5:]

# %% [markdown]
# ## Analyze

# %%
date_from = pd.to_datetime("2021-01-01")
date_to   = pd.to_datetime("2021-12-31")

countries = sorted([
    "Poland",
    "Czechia",
    "Germany",
    "Austria",
    "United Kingdom"
])


# %%
def plot(df):
    fig, ax = plt.subplots(figsize=(16,5))
    df.plot(ax=ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


# %% [markdown]
# ### Plot cases day by day

# %%
cases_day_by_day = df.loc[(countries, slice(date_from, date_to)), "Cases"]
cases_day_by_day.groupby(level="Country/Region").head()

# %%
plot(cases_day_by_day.unstack(level="Country/Region"))


# %% [markdown]
# ### Plot cases week-by-week

# %%
def week_start(dt):
    return dt - timedelta(days=dt.weekday())


# %%
cases_week_by_week = (
    df.loc[(countries, slice(week_start(date_from), date_to)), "Cases"]
    .groupby(level="Country/Region")
    .resample("1W", level="Date")
    .sum()
)
cases_week_by_week.groupby(level="Country/Region").head()

# %%
plot(cases_week_by_week.unstack(level="Country/Region"))


# %% [markdown]
# ### Plot moving average of cases

# %%
def moving_average(df, window_days, date_from, date_to):
    return (
        df.loc[(slice(None), slice(date_from - timedelta(days=window_days-1), date_to)), "Cases"]
        .groupby(level="Country/Region")
        .apply(lambda s: s.rolling(7).sum())
        .loc[(countries, slice(date_from, date_to))]
    )


# %%
cases_moving_average = moving_average(df.loc[countries], 7, date_from, date_to)
cases_moving_average.groupby(level="Country/Region").head(10)

# %%
plot(cases_moving_average.unstack(level="Country/Region"))


# %% [markdown]
# ### Compare 2020 and 2021

# %%
def moving_average_year_to_year(df, window_days, years):
    return pd.concat(
        moving_average(df.loc[countries], window_days, pd.to_datetime(f"{year}-04-01"), pd.to_datetime(f"{year}-11-15"))
        .reset_index()
        .assign(**{"Year": year, "Day": lambda df: df["Date"].dt.strftime("%m-%d")})
        .drop(columns=["Date"])
        for year in years
    ).set_index(["Country/Region", "Year", "Day"]).sort_index()


# %%
cases_year_to_year = moving_average_year_to_year(df.loc[countries], window_days=7, years=[2020, 2021])
cases_year_to_year.groupby(["Country/Region", "Year"]).head(5)


# %%
def plot_cases_year_to_year(df):
    countries, years, days = df.index.levels
    fig, axs = plt.subplots(nrows=len(countries), ncols=1, figsize=(16, 24))
    for ax, country in zip(axs, countries):
        for year in years:
            cases = df.loc[(country, year), "Cases"]
            ax.plot(cases.index, cases.values, label=year)
        ax.set_title(country)
        ax.set_xticks([day for day in days if day.endswith("-01")])
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))


# %%
plot_cases_year_to_year(cases_year_to_year)
