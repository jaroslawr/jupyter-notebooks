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
# # Pandas reference

# %% [markdown]
# Quick reference on getting common data processing tasks done with Pandas.

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Import libraries

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ### Show more data in dataframes

# %%
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 200

# %% [markdown]
# ### Set floating point precision

# %%
pd.options.display.precision = 3


# %% [markdown]
# ## Datasets

# %% [markdown]
# ### USD exchange rates (yearly averages)

# %%
def usd_exchange_rates_df():
    return pd.DataFrame(
        columns=("Year", "Currency", "Currency/USD", "USD/Currency"),
        data=[
            [pd.to_datetime("2016-12-31"), "EUR", 1.064, 0.940],
            [pd.to_datetime("2017-12-31"), "EUR", 1.083, 0.923],
            [pd.to_datetime("2018-12-31"), "EUR", 1.179, 0.848],
            [pd.to_datetime("2019-12-31"), "EUR", 1.120, 0.893],
            [pd.to_datetime("2020-12-31"), "EUR", 1.140, 0.877],
            [pd.to_datetime("2016-12-31"), "GBP", 1.299, 0.770],
            [pd.to_datetime("2017-12-31"), "GBP", 1.238, 0.808],
            [pd.to_datetime("2018-12-31"), "GBP", 1.333, 0.750],
            [pd.to_datetime("2019-12-31"), "GBP", 1.276, 0.784],
            [pd.to_datetime("2020-12-31"), "GBP", 1.284, 0.779],
        ]
    )


# %% [markdown]
# ## Filtering

# %% [markdown]
# ### Filter with []

# %% [markdown]
# Select rows with `[]`:

# %%
df = usd_exchange_rates_df()
df[df["Currency"] == "EUR"]

# %% [markdown]
# Select a single column as a `pd.Series` with `[]`:

# %%
df = usd_exchange_rates_df()
df["Currency/USD"]

# %% [markdown]
# Select one or more columns as a `pd.DataFrame` by passing a list to `[]`:

# %%
df = usd_exchange_rates_df()
df[["Currency/USD"]]

# %% [markdown]
# Selection of rows and of columns can be combined:

# %%
df = usd_exchange_rates_df()
df[df["Currency"] == "EUR"]["Currency/USD"]

# %% [markdown]
# Note that chaining `[]` does not work for the purpose of modifying or inserting data:

# %%
df = usd_exchange_rates_df()
df[df["Currency"] == "EUR"]["Currency/USD"] = 5

# %% [markdown]
# `df[][]=` translates to a `df.__getitem__()` call on the data frame and then a `.__setitem__()` call on the resulting object. The problem is that the `df.__getitem__()` call might return either a view or a copy of the dataframe, so the dataframe might or might not be modified.
#
# Instead, `df.loc[]` can be used to select rows and columns at the same time. `df.loc[]` will return a view or a copy just like `df[]`, but `df.loc[]=` is just a single method call on the `loc` attribute of the original dataframe, free of the ambiguity of `[][]=`, so that it will always correctly modify the dataframe.

# %% [markdown]
# ### Filter with loc[]

# %% [markdown]
# Select rows:

# %%
df = usd_exchange_rates_df()
df.loc[df["Currency"] == "EUR"]

# %% [markdown]
# Select a single column as a `pd.Series`:

# %%
df = usd_exchange_rates_df()
df.loc[:, "Currency/USD"]

# %% [markdown]
# Select one or more columns as a `pd.DataFrame`:

# %%
df = usd_exchange_rates_df()
df.loc[:, ["Currency/USD"]]

# %% [markdown]
# Modify a subpart of a dataframe:

# %%
df = usd_exchange_rates_df()
df.loc[df["Currency"] == "EUR", "Currency/USD"] = 2
df.loc[df["Currency"] == "GBP", "USD/Currency"] = 0.5
df

# %% [markdown]
# `loc` is also used to filter rows and columns using a multi-index. In this case, always both the row and column filter need to be passed as arguments:

# %%
df = usd_exchange_rates_df().set_index(["Currency", "Year"])
df.head(1)

# %%
df.loc[("EUR", pd.to_datetime("2017-12-31")), :]

# %% [markdown]
# The elements of the tuple can be `slice()` objects or lists:

# %%
df.loc[(["EUR", "GBP"], slice(pd.to_datetime("2017-12-31"), pd.to_datetime("2019-12-31"))), :]

# %% [markdown]
# ### Boolean masks for [] and loc[]

# %% [markdown]
# Boolean masks can be formed with `&`, `|` and `~` (negation) and passed to `[]` and to `loc[]`. Conditions have to be enclosed in parenthesis since `&` and `|` have higher priority in Python than operators like `>=`:

# %%
df = usd_exchange_rates_df()
df[(df["Year"] >= pd.to_datetime("2018-12-31")) &
   (df["Year"] <= pd.to_datetime("2020-12-31"))]

# %% [markdown]
# The condition inside `[]` translates to a boolean vector:

# %%
((df["Year"] >= pd.to_datetime("2018-12-31")) &
 (df["Year"] <= pd.to_datetime("2020-12-31")))

# %% [markdown]
# Use `isin()` series method for subset selection:

# %%
df = usd_exchange_rates_df()
df[df["Year"].isin([
    pd.to_datetime("2018-12-31"),
    pd.to_datetime("2019-12-31"),
    pd.to_datetime("2020-12-31")
])]

# %% [markdown]
# ## Grouping

# %% [markdown]
# ### Reduce group-by-group and series-by-series with agg()

# %% [markdown]
# `df.groupby().agg(func)` will call `func(series)` once for each series of every group.
#
# `func` should return a scalar.

# %%
df = usd_exchange_rates_df()
df.groupby("Currency").agg(np.mean)

# %% [markdown]
# Multiple aggregations can be specified:

# %%
df = usd_exchange_rates_df()
df.groupby("Currency")[["Currency/USD", "USD/Currency"]].agg([np.mean, np.var])

# %% [markdown]
# Use keyword arguments to rename the resulting columns:

# %%
df = usd_exchange_rates_df()
df.groupby("Currency")[["Currency/USD", "USD/Currency"]].agg(
    avg_cur2usd=("Currency/USD", np.mean),
    avg_usd2cur=("USD/Currency", np.mean),
)

# %% [markdown]
# The last type of agg() aggregation has slightly different syntax when dealing with a single series:

# %%
df = usd_exchange_rates_df()
df.groupby("Currency")["Currency/USD"].agg(average=np.mean)

# %% [markdown]
# ### Reduce group-by-group with apply

# %% [markdown]
# `df.groupby().apply(func)` will call `func(group)` once for each group, where `group` is a dataframe containing the rows within each group.
#
# `func` can return:
# - a scalar - making the result of `apply()` a series
# - a series - making the result of `apply()` a series
# - a dataframe - making the result of `apply()` a dataframe

# %%
df = usd_exchange_rates_df()
df.groupby("Currency")[["Currency/USD", "USD/Currency"]].apply(lambda df: df.mean())

# %% [markdown]
# ### Transform rows one-by-one with transform

# %% [markdown]
# `df.groupby().transform(func)` will call `func(series_in_group)` once for each series in each group. In contrast to `apply()`, the result of `transform()` is of the same dimensions as the original dataframe.
#
# `func(series_in_group)` should either return a series of the same dimensions as `series_in_group` or a scalar, in which case pandas will take care of making a series of length `len(series_in_group)` out of it.

# %%
df = usd_exchange_rates_df()
df.groupby("Currency")[["Currency/USD", "USD/Currency"]].transform(lambda df: df.mean())
