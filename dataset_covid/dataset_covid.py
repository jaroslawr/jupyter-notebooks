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

import pycountry

from datetime import datetime, timedelta

# %%
plt.style.use("ggplot")

# %%
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 200


# %% [markdown]
# ## Import and clean up data

# %%
def read_csv(file, response):
    return (
        pd.read_csv(file)
        # Remove unused columns
        .drop(["Lat", "Long"], axis=1)
        # Rename the columns
        .rename(columns={"Country/Region": "country", "Province/State": "province"})
        # Make the date a column
        .melt(id_vars=["province", "country"], var_name="date", value_name=response)
        # Convert the date from string to a real date
        .assign(**{"date": lambda df: pd.to_datetime(df["date"])})
        # Aggregate provinces
        .groupby(["country", "date"])
        .sum()
        # Convert cumulative totals to daily increments
        .groupby(level="country")
        .diff()
    )


# %%
cases_df = read_csv("time_series_covid19_confirmed_global.csv", "cases")

# %%
cases_df.head(5)

# %%
deaths_df = read_csv("time_series_covid19_deaths_global.csv", "deaths")

# %%
deaths_df.head(5)

# %% [markdown]
# Read population data from https://data.worldbank.org/indicator/SP.POP.TOTL?view=chart:

# %%
pop_df = (
    pd.read_csv("worldbank_pop.csv")
    .loc[:, ["Country Code", "2020"]]
    .assign(pop_mln=lambda df: df["2020"]/1_000_000)
    .rename(columns={"Country Code": "country_code"})
    .loc[:, ["country_code", "pop_mln"]]
    .set_index("country_code")
    .sort_index()
)

# %%
pop_df.head(3)

# %% [markdown]
# Map country names from the John-Hopkins University Covid datasets to three letter ISO codes using pycountry:

# %%
country_names = cases_df.index.levels[0].values
country_names_to_codes = {}
country_names_unmapped = []
for country_name in country_names:
    try:
        search_results = pycountry.countries.search_fuzzy(country_name)
        country_names_to_codes[country_name] = search_results[0].alpha_3
    except LookupError:
        country_names_unmapped.append(country_name)

country_names_to_codes = pd.Series(country_names_to_codes, name="country_code")
country_names_to_codes.head(5)

# %% [markdown]
# Country names that could not be mapped:

# %%
country_names_unmapped

# %% [markdown]
# Join all the dataframes:

# %%
df = (
    cases_df.join(deaths_df)
    .join(country_names_to_codes, on="country", how="inner")
    .join(pop_df, on="country_code", how="inner")
    .drop(columns="country_code")
    .loc[lambda df: df["pop_mln"].notna()]
    .assign(cases_per_mln=lambda df: df["cases"] / df["pop_mln"],
            deaths_per_mln=lambda df: df["deaths"] / df["pop_mln"])
)

# %%
df.groupby("country").tail(1).head(30)

# %% [markdown]
# Countries where the population could not be joined and are hence not included in the final dataframe:

# %%
set(cases_df.index.levels[0].values) - set(df.index.levels[0].values)

# %% [markdown]
# ## Analyze

# %%
date_from = pd.to_datetime("2021-06-01")
date_to   = pd.to_datetime("2021-12-31")

countries = sorted([
    "US",
    "Germany",
    "United Kingdom",
    "Russia",
    "Poland",
    "Ukraine",
    "Czechia"
])


# %%
def plot(df, plot_title):
    fig, ax = plt.subplots(figsize=(16,5))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title(plot_title)


# %% [markdown]
# ### Moving average of cases and deaths

# %%
def moving_average(df, date_from, date_to):
    window_in_days = 7
    return (
        df.loc[(slice(None), slice(date_from - timedelta(days=window_in_days-1), date_to)), :]
        .groupby(level="country")
        .apply(lambda s: s.rolling(window_in_days).mean())
        .loc[(countries, slice(date_from, date_to)), :]
    )


# %%
moving_average_df = moving_average(df.loc[countries], date_from, date_to)
moving_average_df.groupby(level="country").tail(2)

# %%
plot(moving_average_df.loc[:, "cases_per_mln"].unstack(level="country"), "Cases / 1M Citizens")

# %%
plot(moving_average_df.loc[:, "deaths_per_mln"].unstack(level="country"), "Deaths / 1M Citizens")
