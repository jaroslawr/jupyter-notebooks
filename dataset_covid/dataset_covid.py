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
country_names = cases_df.index.unique(level="country")
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
    .sort_index()
)

# %%
df.groupby("country").tail(1).head(30)

# %% [markdown]
# Countries where the population could not be joined and are hence not included in the final dataframe:

# %%
set(cases_df.index.unique(level="country")) - set(df.index.unique(level="country"))

# %% [markdown]
# ## Analyze

# %%
countries = sorted([
    "US",
    "Germany",
    "United Kingdom",
    "Russia",
    "Poland",
    "Ukraine",
    "Czechia"
])


# %% [markdown]
# ### Moving average of cases and deaths

# %%
def moving_average(df):
    return df.groupby(level="country").apply(lambda s: s.rolling(7).mean())


# %%
def plot_moving_average(df, countries, date_from, date_to, response, plot_title):
    plot_df = (
        moving_average(df)
        .loc[(countries, slice(date_from, date_to)), response]
        .unstack(level="country")
    )

    fig, ax = plt.subplots(figsize=(16,5))
    for col in plot_df.columns:
        ax.plot(plot_df.index, plot_df[col], label=col)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title(plot_title)


# %%
plot_moving_average(
    df=df,
    countries=countries,
    date_from=pd.to_datetime("2021-06-01"),
    date_to=pd.to_datetime("2021-12-31"),
    response="cases_per_mln",
    plot_title="Cases / 1M Citizens"
)

# %%
plot_moving_average(
    df=df,
    countries=countries,
    date_from=pd.to_datetime("2021-06-01"),
    date_to=pd.to_datetime("2021-12-31"),
    response="deaths_per_mln",
    plot_title="Deaths / 1M Citizens"
)


# %% [markdown]
# ### Compare 2020 and 2021

# %%
def moving_average_with_1y_shift(df):
    ma = moving_average(df)
    shifted_ma = (
        ma.loc[:, ["cases_per_mln", "deaths_per_mln"]]
        .groupby(level="country")
        .shift(365)
        .rename(columns=lambda c: f"{c}_1y_ago")
    )
    return ma.join(shifted_ma)


# %%
def plot_moving_average_with_1y_shift(df, countries, date_from, date_to):
    plot_df = (
        moving_average_with_1y_shift(df)
        .loc[(countries, slice(date_from, date_to)), :]
    )

    fig, axs = plt.subplots(nrows=len(countries), ncols=2, figsize=(16, 32), sharex=True, sharey="col")
    plt.subplots_adjust(hspace=0.45)

    for axs_row, country in zip(axs, plot_df.index.unique(level="country")):
        for ax, response in zip(axs_row, ["cases_per_mln", "deaths_per_mln"]):
            country_df = plot_df.loc[country, :]
            ax.plot(country_df.index, country_df[response], label="2021")
            ax.plot(country_df.index, country_df[f"{response}_1y_ago"], linestyle="dashed", label="2020")
            ax.set_title(f"{country} - {response}")
            ax.legend(loc="center", ncol=2, bbox_to_anchor=(0.5, -0.2))
            ax.tick_params(axis="x", reset=True)


# %%
plot_moving_average_with_1y_shift(
    df=df,
    countries=countries,
    date_from=pd.to_datetime("2021-01-01"),
    date_to=pd.to_datetime("2021-12-31")
)
