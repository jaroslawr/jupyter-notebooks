# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
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

import datetime

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
df.drop(["Lat", "Long"], axis=1, inplace=True)

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
   (df["Date"] == datetime.datetime(2021, 1, 1))]

# %%
df = df.groupby(["Country/Region", "Date"], as_index=False).sum()

# %%
df[(df["Country/Region"] == "United Kingdom") &
   (df["Date"] == datetime.datetime(2021, 1, 1))]

# %% [markdown]
# ## Convert cumulative total to daily new cases

# %%
df[(df["Country/Region"] == "Poland") & 
   (df["Date"] >= datetime.datetime(2021, 1, 1)) & 
   (df["Date"] <= datetime.datetime(2021, 1, 5))]

# %%
df[["Cases"]] = df[["Country/Region", "Cases"]].groupby("Country/Region").diff()

# %%
df[(df["Country/Region"] == "Poland") & 
   (df["Date"] >= datetime.datetime(2021, 1, 1)) & 
   (df["Date"] <= datetime.datetime(2021, 1, 5))]

# %% [markdown]
# ## Analyze the data

# %%
date_from = datetime.datetime(2021, 1, 1)
date_to = datetime.datetime(2021, 4, 22)
countries = ["Poland", "Czechia", "Germany", "Austria"]


# %%
def plot(countries, date_from, date_to, w, h):
    plot_df = df[df["Country/Region"].isin(countries) & 
                 (df["Date"] >= date_from) &
                 (df["Date"] <= date_to)]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(w, h)

    for country, group_df in plot_df.groupby("Country/Region"):
        ax.plot(group_df["Date"], group_df["Cases"], label=country)

    fig.legend(loc="center right", bbox_to_anchor=(1.02, 0.5))

plot(countries=countries, date_from=date_from, date_to=date_to, w=16, h=6)
