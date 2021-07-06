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
# # Pandas reference

# %% [markdown]
# Quick reference on getting common data processing tasks done with Pandas.

# %% [markdown]
# ## Environment setup

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
pd.options.display.precision = 2


# %% [markdown]
# ## Example dataframes

# %%
def example_df():
    return pd.DataFrame(
        columns=("Cat", "Val1", "Val2"),
        data=[
            ["C1", 1.0, 2.0],
            ["C1", 3.0, 4.0],
            ["C2", 5.0, 6.0],
            ["C2", 7.0, 8.0],
        ]
    )


# %% [markdown]
# ## Selecting

# %% [markdown]
# ### Selecting with .[]

# %% [markdown]
# You can select rows with `.[]`:

# %%
df = example_df()
df[df["Cat"] == "C1"]

# %% [markdown]
# You can also select columns with `.[]`:

# %%
df = example_df()
df["Cat"]

# %% [markdown]
# You can select both rows and columns by chaining the `.[]`:

# %%
df = example_df()
df[df["Cat"] == "C1"]["Val1"]

# %% [markdown]
# This last usage leads to pitfalls, for example when trying to modify the values in the cells selected in this way. `.loc[]` is the preferred way to select both rows and columns.

# %% [markdown]
# ### Selecting with .loc

# %% [markdown]
# However, `.[]` does not work when you want to select both rows and columns with the purpose of modifying or inserting data:

# %%
df = example_df()
df[df["Cat"] == "C1"]["Val1"] = 5

# %% [markdown]
# `df[][]=` translates to a `df.__getitem__()` call on the data frame and then a `.__setitem__()` call on the resulting object. What is problematic is that `df.__getiem__()` might return either a view or a copy of the dataframe.

# %% [markdown]
# Instead, `.loc[]` can be used for selecting both rows and columns, in particular for modifying or inserting data:

# %%
df = example_df()
df.loc[df["Cat"] == "C1", "Val3"] = 9
df.loc[df["Cat"] == "C2", "Val3"] = 10
df

# %% [markdown]
# `df.loc[]` returns either a view or a copy just like `df[]`, but `df.loc[]=` is just a single method call on the `.loc` attribute of the original dataframe, free of the ambiguity of `.[][]=`, so that it will always correctly modify the dataframe.

# %% [markdown]
# ### Boolean masks

# %% [markdown]
# You can form boolean masks with `&`, `|` and `~` (negation) and pass them to `.[]` and `.loc[]`. Conditions have to be enclosed in parenthesis since `&` and `|` have higher priority in Python than operators like `>=`:

# %%
df = example_df()
df[(df["Val2"] >= 4.0) & (df["Val2"] <= 6.0)]

# %% [markdown]
# Use `.isin()` series method for subset selection:

# %%
df = example_df()
df[df["Val2"].isin([4.0, 8.0])]

# %% [markdown]
# ## Grouping

# %% [markdown]
# ### agg: reduce group-by-group and series-by-series

# %% [markdown]
# `df.groupby().agg(func)` will call `func(series)` once for each series of every group.
#
# `func` should return a scalar.

# %%
df = example_df()
df.groupby("Cat").agg(np.mean)

# %% [markdown]
# Multiple aggregations can be specified:

# %%
df = example_df()
df.groupby("Cat").agg([np.mean, np.var])

# %% [markdown]
# Use keyword arguments to rename the resulting columns:

# %%
df = example_df()
df.groupby("Cat").agg(val1_mean=("Val1", np.mean), val2_mean=("Val2", np.mean))

# %% [markdown]
# The last type of agg() aggregation has slightly different syntax when dealing with a single series:

# %%
df = example_df()
df.groupby("Cat")["Val1"].agg(val1_mean=np.mean, val1_var=np.var)

# %% [markdown]
# ### apply: reduce group-by-group

# %% [markdown]
# `df.groupby().apply(func)` will call `func(group)` once for each group, where `group` is a dataframe containing the rows within each group.
#
# `func` can return:
# - a scalar - making the result of `apply()` a series
# - a series - making the result of `apply()` a series
# - a dataframe - making the result of `apply()` a dataframe

# %%
df = example_df()
df.groupby("Cat").apply(lambda df: df.mean())

# %% [markdown]
# ### transform: transform rows one-by-one

# %% [markdown]
# `df.groupby().transform(func)` will call `func(series_in_group)` once for each series in each group. In contrast to `apply()`, the result of `transform()` is of the same dimensions as the original dataframe.
#
# `func(series_in_group)` should either return a series of the same dimensions as `series_in_group` or a scalar, in which case pandas will take care of making a series of length `len(series_in_group)` out of it.

# %%
df = example_df()
df.groupby("Cat").transform(lambda df: df.mean())
