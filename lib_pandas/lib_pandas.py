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
# ## Grouping

# %% [markdown]
# Dataframe for examples that follow:

# %%
df = pd.DataFrame(
    columns=("Cat", "Val1", "Val2"),
    data=[
        ["C1", 1.0, 2.0],
        ["C1", 3.0, 4.0],
        ["C2", 5.0, 6.0],
        ["C2", 7.0, 8.0],
    ]
)

# %%
df

# %% [markdown]
# ### agg: reduce group-by-group and series-by-series

# %% [markdown]
# `df.groupby().agg(func)` will call `func(series)` once for each series of every group.
#
# `func` should return a scalar.

# %%
df.groupby("Cat").agg(np.mean)

# %% [markdown]
# Multiple aggregations can be specified:

# %%
df.groupby("Cat").agg([np.mean, np.var])

# %% [markdown]
# Use keyword arguments to rename the resulting columns:

# %%
df.groupby("Cat").agg(val1_mean=("Val1", np.mean), val2_mean=("Val2", np.mean))

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
df.groupby("Cat").apply(lambda df: df.mean())

# %% [markdown]
# ### transform: transform rows one-by-one

# %% [markdown]
# `df.groupby().transform(func)` will call `func(series_in_group)` once for each series in each group. In contrast to `apply()`, the result of `transform()` is of the same dimensions as the original dataframe.
#
# `func(series_in_group)` should either return a series of the same dimensions as `series_in_group` or a scalar, in which case pandas will take care of making a series of length `len(series_in_group)` out of it.

# %%
df.groupby("Cat").transform(lambda df: df.mean())
