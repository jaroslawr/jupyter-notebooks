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
   (df["Date"] >= datetime(2021, 1, 1)) & 
   (df["Date"] <= datetime(2021, 1, 5))]

# %%
df[["Cases"]] = df[["Country/Region", "Cases"]].groupby("Country/Region").diff()

# %%
df[(df["Country/Region"] == "Poland") & 
   (df["Date"] >= datetime(2021, 1, 1)) & 
   (df["Date"] <= datetime(2021, 1, 5))]


# %% [markdown]
# ## Analyze the data

# %%
def to_start_of_week(date):
    return date - timedelta(days=date.weekday())


# %%
date_from = to_start_of_week(datetime(2021, 1, 1))
date_to = to_start_of_week(datetime(2021, 4, 22))
countries = ["Poland", "Czechia", "Germany", "Austria"]

# %%
df = df.loc[df["Country/Region"].isin(countries) & 
            (df["Date"] >= date_from) &
            (df["Date"] <= date_to)]

# %%
df.head(3)

# %% [markdown]
# ### Day by day plot

# %%
sns.relplot(data=df, x="Date", y="Cases", hue="Country/Region", kind="line", aspect=3)


# %% [markdown]
# ### Week by week plot

# %%
def week_by_week_plot(df):
    plot_df = df.copy()
    plot_df["Week"] = plot_df["Date"].apply(to_start_of_week)
    plot_df = plot_df[(plot_df["Date"] >= to_start_of_week(date_from)) &
                      (plot_df["Date"] < to_start_of_week(date_to))]
    plot_df = plot_df.groupby(["Country/Region", "Week"]).sum().reset_index()
    sns.relplot(data=plot_df, x="Week", y="Cases", hue="Country/Region", kind="line", marker="o", aspect=3)

week_by_week_plot(df)
