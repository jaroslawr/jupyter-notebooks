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
# # Data analysis: Dow Jones stock prices over time

# %% [markdown] tags=[]
# Data analysis of Dow Jones stock prices dataset from:
#
# http://archive.ics.uci.edu/ml/datasets/Dow+Jones+Index

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Setup

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## Read and clean data

# %% [markdown]
# ### Read CSV

# %%
df = pd.read_csv("dow_jones_index.data")

# %% [markdown]
# ### Examine results

# %%
df.head(5)

# %%
df.dtypes

# %% [markdown]
# ### Convert date column to date datatype

# %%
df.head(1)["date"]

# %%
df["date"] = pd.to_datetime(df["date"])

# %%
df.head(1)["date"]

# %%
df.dtypes


# %% [markdown]
# ### Convert prices in dollars to floats

# %%
def parse_dollar_price(dollar_price):
    return np.float64(dollar_price[1:]) if dollar_price[0] == '$' else np.NaN

dollars_cols = ["open", "high", "low", "close", "next_weeks_open", "next_weeks_close"]
df[dollars_cols] = df[dollars_cols].applymap(parse_dollar_price)

# %%
df.head(5)

# %%
df.dtypes

# %% [markdown]
# ## Analyze and plot

# %%
df.head(3)


# %%
def plot(w, h):
    fig, ax = plt.subplots()
    fig.set_size_inches(w, h)

    for stock, stock_df in df.groupby("stock"):
        ax.plot(stock_df["date"], stock_df["close"], label=stock)

    fig.legend(loc='center right')

plot(16, 8)

# %%
df[["stock", "close"]].groupby("stock").agg(
    p1=("close", lambda s: s.quantile(0.01)),
    p5=("close", lambda s: s.quantile(0.05)),
    p25=("close", lambda s: s.quantile(0.25)),
    p50=("close", lambda s: s.quantile(0.5)),
    p75=("close", lambda s: s.quantile(0.75)),
    p95=("close", lambda s: s.quantile(0.95)),
    p99=("close", lambda s: s.quantile(0.99))
).sort_values(by="p50", ascending=False).head(5)
