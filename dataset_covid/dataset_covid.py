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

# %%
def read_csv(file, response):
    return (
        pd.read_csv(file)
        # Remove unused columns
        .drop(["Lat", "Long"], axis=1)
        # Make the date a column
        .melt(id_vars=["Province/State", "Country/Region"], var_name="Date", value_name=response)
        # Convert the date from string to a real date
        .assign(**{"Date": lambda df: pd.to_datetime(df["Date"])})
        # Aggregate provinces
        .groupby(["Country/Region", "Date"])
        .sum()
        # Convert cumulative totals to daily increments
        .groupby(level="Country/Region")
        .diff()
    )


# %%
cases_df = read_csv("time_series_covid19_confirmed_global.csv", "Cases")

# %%
cases_df.head(5)

# %%
deaths_df = read_csv("time_series_covid19_deaths_global.csv", "Deaths")

# %%
deaths_df.head(5)

# %%
df = cases_df.join(deaths_df)

# %%
df.head(5)

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
    cases_df.loc[(countries, slice(week_start(date_from), date_to)), "Cases"]
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
def moving_average(df, date_from, date_to):
    window_days = 7
    return (
        df.loc[(slice(None), slice(date_from - timedelta(days=window_days-1), date_to)), :]
        .groupby(level="Country/Region")
        .apply(lambda s: s.rolling(7).sum())
        .loc[(countries, slice(date_from, date_to)), :]
    )


# %%
cases_moving_average = moving_average(df.loc[countries], date_from, date_to)
cases_moving_average.groupby(level="Country/Region").tail(10)

# %%
plot(cases_moving_average.loc[:, "Cases"].unstack(level="Country/Region"))


# %% [markdown]
# ### Compare 2020 and 2021

# %%
def moving_average_year_to_year(df, date_from, date_to):
    levels = [countries, pd.date_range(date_from, date_to)]
    index = pd.MultiIndex.from_product(levels, names=df.index.names)

    return (
        moving_average(df.loc[countries], date_from, date_to)
        .reindex(index)
        .reset_index()
        .assign(**{
            "Year": lambda df: df["Date"].dt.year,
            "Day": lambda df: df["Date"].dt.strftime("%m-%d")
        })
        .drop(columns=["Date"])
        .set_index(["Country/Region", "Year", "Day"])
        .sort_index()
    )


# %%
cases_year_to_year = moving_average_year_to_year(df.loc[countries], pd.to_datetime("2020-01-01"), pd.to_datetime("2021-12-31"))
cases_year_to_year.head(5)


# %%
def plot_cases_year_to_year(df):
    countries, years, days = df.index.levels
    
    fig, axs = plt.subplots(nrows=len(countries), ncols=2, figsize=(16, 24))
    plt.subplots_adjust(hspace=0.45)
    
    for row, country in zip(axs, countries):
        for ax, response in zip(row, ["Cases", "Deaths"]):
            for year in years:
                cases = df.loc[(country, year)]
                ax.plot(cases.index, cases[response], label=year)

            ax.set_title(f"{country} - {response}")
            ax.set_xticks([day for day in days if day.endswith("-01")])
            ax.legend(loc="center", ncol=2, bbox_to_anchor=(0.5, -0.2))


# %%
plot_cases_year_to_year(cases_year_to_year)
