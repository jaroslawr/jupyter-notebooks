# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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

# %% [markdown]
# Data from https://www.irs.gov/individuals/international-taxpayers/yearly-average-currency-exchange-rates:

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
# ## Series, dataframes and indexes

# %% [markdown]
# The basic pandas data type is a series which is a single list of data points:

# %%
series = pd.Series(["a", "b", "c", "d"])
series

# %% [markdown]
# The data points have an associated index: a set of labels for the data points, displayed in the left column of the output above. The default index consists simply of the position of the point in the series. Here is a series with an explicit index:

# %%
series = pd.Series(["e", "f", "g", "h"], index=["a", "b", "c", "d"])
series

# %% [markdown]
# The elements can be accessed using the index values via the `loc[]` method:

# %%
series = pd.Series(["e", "f", "g", "h"], index=["a", "b", "c", "d"])
series.loc["a"]

# %% [markdown]
# Selection by position in the list of data points is always possible using `iloc[]` regardless of what the index is:

# %%
series = pd.Series(["e", "f", "g", "h"], index=["a", "b", "c", "d"])
series.iloc[0]

# %% [markdown]
# The key in the index can consists of multiple values, in which case the index is called a *multi-index*. Each value in the tuple is called a *level* of the index and each level can optionally have a name:

# %%
index = pd.MultiIndex.from_tuples([("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")], names=["l1", "l2"])
series = pd.Series(["e", "f", "g", "h"], index=index)
series

# %% [markdown]
# A dataframe is a collection of series and it has two indexes: a row index, mapping a row key to a row, and a column index, mapping a column key to a series.

# %% [markdown]
# ## Selecting rows and columns

# %% [markdown]
# ### Select with []

# %% [markdown]
# Select rows with `[]`:

# %%
df = usd_exchange_rates_df()
df[df["Currency"] == "EUR"]

# %% [markdown]
# `[]` will accept a callable as argument for cases where a reference to the dataframe is not available, for example when chaining method calls on the dataframe:

# %%
df = usd_exchange_rates_df()
df[lambda df: df["Year"] >= pd.to_datetime("2018-12-31")][lambda df: df["Currency"] == "EUR"]

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
# ### Select with loc[]

# %% [markdown]
# Select rows:

# %%
df = usd_exchange_rates_df()
df.loc[df["Currency"] == "EUR"]

# %% [markdown]
# `loc[]` will accept a callable as argument for cases where a reference to the dataframe is not available, for example when chaining method calls on the dataframe:

# %%
df = usd_exchange_rates_df()
(df.loc[lambda df: df["Year"] >= pd.to_datetime("2018-12-31")]
 .loc[lambda df: df["Currency"] == "EUR"])

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
# ### Select with loc[] and a multi-index

# %% [markdown]
# When using `loc[]` with multi-index both the row and column filter need to be passed as arguments:

# %%
df = usd_exchange_rates_df().set_index(["Currency", "Year"])
df.loc[("EUR", pd.to_datetime("2017-12-31")), :]

# %% [markdown]
# The elements of the tuple can be lists or `slice()` objects:

# %%
df = usd_exchange_rates_df().set_index(["Currency", "Year"])
df.loc[(["EUR", "GBP"], slice(pd.to_datetime("2017-12-31"), pd.to_datetime("2019-12-31"))), :]

# %% [markdown]
# To filter only on first N levels of the multi-index, simply omit the criteria for all remaining levels:

# %%
df = usd_exchange_rates_df().set_index(["Currency", "Year"])
df.loc["EUR", :]

# %% [markdown]
# To filter only on middle or last N levels of the multi-index, use `slice(None)` to select everything at the upper levels (`slice(None)` is just the equivalent of `:`, which can not be used as an element in a tuple):

# %%
df = usd_exchange_rates_df().set_index(["Currency", "Year"])
df.loc[(slice(None), pd.to_datetime("2017-12-31")), :]

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
df = usd_exchange_rates_df()
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
# ### Reduce group-by-group with apply()

# %% [markdown]
# `df.groupby().apply(func)` will call `func(group)` once for each group, where `group` is a dataframe containing the rows within each group. The form of the result depends of the return type of `func`.

# %% [markdown] tags=[]
# **Case 1:** `func` returns a scalar - the result of `apply(func)` is a series indexed by the group key:

# %% tags=[]
df.groupby("Currency")[["Currency/USD"]].apply(lambda df: np.mean(df.values))

# %% [markdown] tags=[]
# **Case 2:** `func` returns a series - the result of `apply(func)` is a dataframe indexed by the group key, with columns given by the index of the returned series:

# %% tags=[]
df = usd_exchange_rates_df()
df.groupby("Currency")[["Currency/USD", "USD/Currency"]].apply(lambda df: df.mean())

# %% [markdown] tags=[]
# **Case 3:** `func` returns a dataframe - the result of `apply(func)` is a dataframe with a multi-index and with same columns as the dataframe returned by `func`. The multi-index consists of a group key level concatenated with levels of the index of the dataframes returned by `func`:

# %% tags=[]
df = usd_exchange_rates_df()
df.groupby("Currency").apply(lambda df: df.drop(columns=["Currency"]).set_index(["Year"]).rolling(3).mean().dropna())

# %% [markdown]
# ### Reduce group-by-group series-by-series with agg()

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
# ### Transform rows one-by-one with transform()

# %% [markdown]
# `df.groupby().transform(func)` will call `func(series_in_group)` once for each series in each group. In contrast to `apply()`, the result of `transform()` is of the same dimensions as the original dataframe.
#
# `func(series_in_group)` should either return a series of the same dimensions as `series_in_group` or a scalar, in which case pandas will take care of making a series of length `len(series_in_group)` out of it.

# %%
df = usd_exchange_rates_df()
df.groupby("Currency")[["Currency/USD", "USD/Currency"]].transform(lambda df: df.mean())

# %% [markdown]
# ## Pivoting and unpivoting

# %% [markdown]
# The USD exchange rates dataframe is neither in fully long format nor in fully wide format: there is one row per each year+currency pair, but there are two different "observations" stored in two columns: "Currency/USD" and "USD/Currency".
#
# Thus the dataframe can be both:
# - pivoted (widened) so that each currency becomes a separate column
# - unpivoted (lengthened) so that each row is split into two and there is a "Direction" column equal to either "Currency/USD" or "USD/Currency" and a value column with the actual rate

# %% [markdown]
# ### Pivot with multi-index and unstack()

# %%
df = usd_exchange_rates_df().set_index(["Currency", "Year"])
df.unstack(level=0)

# %% [markdown]
# ### Unpivot with multi-index and stack()

# %%
df = usd_exchange_rates_df().set_index(["Currency", "Year"])
df.stack().rename_axis(index={None: "Direction"})

# %% [markdown]
# ### Pivot with pivot()

# %%
df = usd_exchange_rates_df()
df.pivot(index=["Year"], columns=["Currency"], values=["Currency/USD", "USD/Currency"])

# %% [markdown]
# ### Unpivot with melt()

# %% tags=[]
df = usd_exchange_rates_df()
df.melt(id_vars=["Year", "Currency"], var_name="Direction", value_name="Value")
