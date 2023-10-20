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
# # Data analysis with Pandas

# %% [markdown]
# This is a guide to doing data analysis with Pandas written to help understand the most important concepts and usage patterns in a way that sticks with you. I try to make things understandable and memorable by adhering to some principles:
#
# - Identify and explore important general concepts:
#     - the basic Pandas data structures: series, dataframes, indexes and multi-indexes
#     - wide classes of operations on data: selection, grouping, aggregations and transformations, sorting
#     - shapes that datasets come in and how to convert between them: long vs wide
# - Use well chosen examples:
#     - preferrably real datasets
#     - understandable for a broad audience
#     - interesting
#     - distinct from one another in structure
# - Stay concise, avoid minutiae that can be easily looked up in the documentation, provide pointers to references instead
#

# %% [markdown]
# Before we begin we need to import all the libraries we will use. We also change some options that I almost always find useful to change:

# %%
# Import libraries
import numpy as np
import pandas as pd

# Make Pandas less eager to hide data in dataframes with "..."
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100
pd.options.display.max_colwidth = 200

# Sensible floating point precision for basic use cases
pd.options.display.precision = 3

# Short exception tracebacks
# %xmode Plain

# %% [markdown]
# ## Cars dataset

# %% [markdown]
# We will use the mtcars dataset as the default basic example in the following sections. Its contents are easy to understand and interesting and also it is provided in a common simple form: each row describes a single unique unit of observation (a car) and each column is a different measurement for the same unit of observation (miles/gallon, number of cylinders, horsepower etc.):

# %%
CARS = pd.read_csv("mtcars.csv")

def cars():
    # Return a fresh copy every time so that examples do not affect each other
    return CARS.copy()


# %% [markdown]
# ## Exploring datasets

# %% [markdown]
# `.head(n)` returns the first `n` rows, `.tail(n)` the last `n` rows, and `.sample(n)` random `n` rows:

# %%
df = cars()
display(df.head(2))
display(df.tail(2))
display(df.sample(2))

# %% [markdown]
# `info()` prints a summary description of a dataframe that contains a lot of useful information: number of entries in the index (which is equal to the total number of rows), a list of columns along with the datatype of the column and the number of non-null values and overall memory usage of the dataframe. The downside is that all this information is simply printed and nothing is returned:

# %%
df = cars()
df.info()

# %% [markdown]
# `len(df)` returns the number of rows in a dataframe:

# %%
df = cars()
len(df)

# %% [markdown]
# `count()` returns number of rows with non-null value. Since this differs for each column series, when called on a dataframe it returns a series with a single value for each column label:

# %%
df = cars()
df.count()

# %% [markdown]
# `df.dtypes` returns a series indexed by column name that contains the dtype of each column of `df`:

# %%
df = cars()
df.dtypes

# %% [markdown]
# `describe()` returns a dataframe that basic summary statistics and the number of non-null rows for each column:

# %%
df = cars()
df.describe()

# %% [markdown]
# `value_counts()` on the given column series returns a series describing how many rows take each unique value:

# %%
df = cars()
df["cyl"].value_counts()

# %% [markdown]
# `value_counts(normalize=True)` returns a series describing what fraction of all rows have each unique value, rather than the raw counts:

# %%
df = cars()
df["cyl"].value_counts(normalize=True)

# %% [markdown]
# `unique()` returns only the unique value themselves:

# %%
df["cyl"].unique()

# %% [markdown]
# `nunique()` returns the number of unique values:

# %%
df["cyl"].nunique()

# %% [markdown]
# `nlargest(n)` returns `n` largest values and `nsmallest(n)` `n` smallest (`n` defaults to 5):

# %%
df["hp"].nlargest()

# %%
df["hp"].nsmallest()

# %% [markdown]
# To see the cars with largest horse power, use the fact that the index of the series returned by `nlargest` is a subset of the dataframe index and hence can be used to subset the dataframe:

# %%
df.loc[df["hp"].nlargest().index]

# %% [markdown]
# ## Selecting rows and columns

# %% [markdown]
# ### Selecting with []

# %% [markdown]
# Select rows with `[]`:

# %%
df = cars()
df[df["gear"] == 5]

# %% [markdown]
# Select a single column as a `pd.Series` with `[]`:

# %%
df = cars()
df["gear"].head(5)

# %% [markdown]
# Select one or more columns as a `pd.DataFrame` by passing a list to `[]` - note how a dataframe with one column is of different type than a series and is displayed in a different way:

# %%
df = cars()
df[["gear"]].head(5)

# %% [markdown]
# Selection of rows and of columns can be combined:

# %%
df = cars()
df[df["gear"] == 5]["cyl"]

# %% [markdown]
# This form of row and column selection does not work for the purpose of modifying or inserting data:

# %%
df = cars()
df[df["gear"] == 5]["cyl"] = 3

# %% [markdown]
# Lets breakdown how the `df[df["gear"] == 5]["cyl"] = 3` expression translates to method calls on the underlying objects: `df[df["gear"] == 5]` translates to a `df.__getitem__(df["gear"] == 5)` call on the data frame and then the `["cyl"] = 3` part to a `.__setitem__(3)` call on the resulting object. The problem is that the `__getitem__` call might return either a view or a copy of the dataframe, so the original dataframe might or might not be modified.
#
# Instead, `df.loc[]` can be used to select rows and columns at the same time. `df.loc[]` will return a view or a copy just like `df[]`, but `df.loc[]=` is just a single `__setitem__` method call on the object stored in the `loc` attribute of the original dataframe, free of the ambiguity of `[][]=`, so that it will always correctly modify the dataframe.

# %% [markdown]
# ### Selecting with loc[]

# %% [markdown]
# Select rows:

# %%
df = cars()
df.loc[df["gear"] == 5]

# %% [markdown]
# Select a single column as a `pd.Series`:

# %%
df = cars()
df.loc[:, "gear"].head(5)

# %% [markdown]
# Select one or more columns as a `pd.DataFrame`:

# %%
df = cars()
df.loc[:, ["gear"]].head(5)

# %% [markdown]
# Modify a subpart of a dataframe:

# %%
df = cars()
df.loc[df["cyl"] == 6, "hp"] = 200
df.loc[df["cyl"] == 6, :]

# %% [markdown]
# ### Boolean expressions in [] and loc[]

# %% [markdown]
# When selecting rows with `df[df["gear"] == 5]`, `df["gear"] == 5` is a `pd.Series` wrapping a boolean vector:

# %%
df = cars()
(df["gear"] == 5).head(5)

# %% [markdown]
# Boolean operators like `&`, `|` and `~` (negation) can be used on those boolean vectors to represent compound filtering conditions. Individual conditions have to be enclosed in parenthesis since `&` and `|` have higher priority in Python than operators like `==`, `>=`, etc.:

# %%
df = cars()
df[(df["cyl"] >= 4) & (df["cyl"] <= 6) & (df["gear"] == 3)]

# %% [markdown]
# Use `isin()` series method for subset selection:

# %%
df = cars()
df[df["cyl"].isin([4, 6]) & (df["gear"] == 3)]

# %% [markdown]
# ### Selecting with iloc[]

# %% [markdown]
# `iloc[]` returns rows and columns specified using array-like indexes (positive or negative offset of the row in the series or of the rows and optionally columns of the data frame):

# %%
df = cars()
df.iloc[2]

# %%
df = cars()
df.iloc[-2]

# %%
df = cars()
df.iloc[0:3]

# %% [markdown]
# ## Groups and aggregations

# %% [markdown]
# ### Basics of groups

# %% [markdown]
# To form groups you call `.groupby` on a series or on a dataframe, supplying the group key in one of several supported ways.

# %% [markdown]
# In the simplest case you supply the name of the column to use as the group key:

# %%
df.groupby("cyl")

# %% [markdown]
# This `DataframeGroupBy` object is the starting point from which various groupwise operations can be done:

# %% tags=[]
df.groupby("cyl").mean(numeric_only=True)

# %% [markdown]
# You can also select a single series from the dataframe groupby object which results in a `SeriesGroupBy` object:

# %% tags=[]
df.groupby("cyl")["mpg"]

# %% [markdown]
# Aggregations then produce a single value per group and result in a `pd.Series`:

# %% tags=[]
df.groupby("cyl")["mpg"].mean()

# %% [markdown]
# ### Aggregating using predefined aggregations

# %% [markdown]
# The `SeriesGroupBy` and `DataframeGroupBy` objects both support calls like `min()`, `max()`, `mean()`, `std()`, `var()`, `quantile()` etc. that are groupwise versions of the respective series/dataframe operation:

# %%
df = cars()
df.groupby(["cyl", "gear"])["hp"].mean()

# %% [markdown]
# To count the number of rows in each group call `.size()` on the groupby object:

# %%
df = cars()
df.groupby(["cyl", "gear"]).size()

# %% [markdown]
# Confusingly, to get the number of rows in the whole dataframe, you have to call `len(df)` or `len(df.index)`, rather than `df.size()`. In a dataframe, `.size` is a field not a method and it holds the number of cells in the dataframe not the number of rows. For example, to see what fraction of all cars have which setup in terms of number of cylinders and number of gears you call `.size()` on the groupby object returned by `groupby(["cyl", "gear"])`, but divide by `len(df)`:

# %%
df = cars()
df.groupby(["cyl", "gear"]).size() / len(df)

# %% [markdown]
# To count the groups themselves use the `.ngroups` attribute:

# %%
df = cars()
df.groupby(["cyl", "gear"]).ngroups

# %% [markdown]
# To compute a compound expression involving group level aggregates, for example the range of values within each group (group max - group min), reference to the groupby object and reuse it:

# %%
df = cars()
df_groupby = df.groupby(["cyl", "gear"])["hp"]
df_groupby.max() - df_groupby.min()

# %% [markdown]
# ### Aggregating using generic agg()

# %% [markdown]
# `agg(func)` calls `func(series)` once for each `series` of every group. `func` should return a scalar. `func` can be passed in as a function or a function name (a string). The result is a series or a dataframe depending whether aggregation is done on a single series or on a dataframe but also whether one aggregation is done or many. There are also multiple ways of providing arguments specifying the aggregations to do. Hence there are many cases which we now try to outline.

# %% [markdown]
# #### Aggregating a single series

# %% [markdown]
# Simplest case is doing a single aggregation on a single series. The result is a series:

# %%
df = cars()
df.groupby("cyl")["hp"].agg("mean")

# %% [markdown]
# To do multiple aggregations pass a list of functions or function names to `agg`. The result will be a dataframe, since in general there is more than one group each of which becomes a row of the result and for each group we compute more than one aggregation each of which becomes a column of the result:

# %%
df = cars()
df.groupby("cyl")["hp"].agg(["size", "mean", "std"])

# %% [markdown]
# `agg()` can also be called with keyword arguments, in which case the name of the argument specifies the name of the column for the aggregated data in the resulting dataframe. The value of each keyword argument should again be a function or a function name to perform the aggregation:

# %%
df = cars()
df.groupby("cyl")["hp"].agg(count="size", average="mean", stddev="std")

# %% [markdown]
# #### Aggregating a dataframe

# %% [markdown]
# Next case is aggregation of multiple series. When a single aggregation is applied, the result is a simple dataframe whose column names are the same as the columns that were aggregated:

# %%
df = cars()
df.groupby("cyl")[["hp", "wt"]].agg("mean")

# %% [markdown]
# When multiple aggregations are applied to a dataframe, the result is a dataframe with a column multi-index:

# %%
df = cars()
df.groupby("cyl")[["hp", "wt"]].agg(["mean", "std"])

# %% [markdown]
# You can avoid the column multi-index by using the keyword arguments to `agg()`. The name of the keyword argument again specifies the name of the aggregated series in the resulting dataframe, but values of the keyword arguments now have to be tuples of the form `(name_of_column_to_aggregate,aggregation)` where `aggregation` is as always a function or function name:

# %%
df = cars()
df.groupby("cyl")[["hp", "wt"]].agg(
    hp_mean=("hp", "mean"),
    hp_std=("hp", "std"),
    wt_mean=("wt", "mean"),
    wt_std=("wt", "std"),
)

# %% [markdown]
# Finally when aggregating a dataframe there is yet another way of specifying arguments for `agg`: to do different aggregations for different columns you can pass a dict as an argument. The result will be a dataframe and if any columns is aggregated using more than one function, it will have a column multi-index:

# %%
df = cars()
df.groupby("cyl")[["hp", "wt"]].agg({
    "hp": ["mean", "std"],
    "wt": "mean"
})
