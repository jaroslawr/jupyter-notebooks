# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
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
#     - the Pandas data model: series, dataframes, indexes and multi-indexes
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
# ## Pandas data model

# %% [markdown]
# We begin by exploring the basic Pandas data structures: series, indexes and dataframes. Our goal in this section is to show how things fit together in Pandas conceptually, rather than to be comprehensive about the specific features. We use really simple examples, but jumping right into analysis of real datasets without the foundation we develop here is a recipe for much confusion. 

# %% [markdown]
# ### Series and basic indexes

# %% [markdown]
# The basic pandas data type is a series which is a collection of data points. For our first example, we look at the number of points scored in the 2022/2023 NBA season by individual players. The numbers for 5 players who scored the most are as follows:

# %%
points_by_pos = pd.Series([2225, 2183, 2138, 2135, 1959])
points_by_pos

# %% [markdown]
# The right column of the output shows the values in the series.  `dtype: int64` refers to the data type of the values.

# %% [markdown]
# The left column of the output shows labels corresponding to the values in the series. The labels are part of the *index* that is a part of every series. By default the series is indexed simply using position of each value in the series. `.loc[]` call for a series takes an index label as argument and looks up the corresponding value in the series. For the default index it ends up working like basic list indexing (though it does not support negative indexing, since it does label lookup):

# %%
points_by_pos = pd.Series([2225, 2183, 2138, 2135, 1959])
points_by_pos.loc[2]

# %% [markdown]
# The index is what makes a series something more than a simple list. It makes it possible to refer to the values in the series by whatever label is appropriate: by a string, by a date, by a pair of numbers, ... It also plays a role similar to a database index, hence the name: it speeds up operations like joins that have to lookup by label repeatedly.  We can usefully index each number in this series with the name of the player who scored that many points:

# %%
points_by_player = pd.Series({
    "Jayson Tatum": 2225,
    "Joel Embiid": 2183,
    "Luka Dončić": 2138,
    "Shai Gilgeous-Alexander": 2135,
    "Giannis Antetokounmpo": 1959
})
points_by_player

# %% [markdown]
# The index object itself looks like this:

# %%
points_by_player.index

# %% [markdown]
# With the index in place, you can now lookup the score for a player using the label:

# %%
points_by_player.loc["Jayson Tatum"]

# %% [markdown]
# You can get a series filtered down to a single data point instead of a scalar value by enclosing the label in a list:

# %%
points_by_player.loc[["Jayson Tatum"]]

# %% [markdown]
# This form is also one of the ways of selecting a subset of the series:

# %%
points_by_player.loc[["Jayson Tatum", "Luka Dončić"]]

# %% [markdown]
# Lookup by label makes series appear similar to a Python `dict` and in some ways you can use series like one:

# %%
"Jayson Tatum" in points_by_player

# %%
points_by_player.keys()

# %%
list(points_by_player.items())

# %% [markdown]
# The labels in the index are stored in a particular order that does not need to follow any natural ordering of the labels, making slice selection possible:

# %%
points_by_player.loc["Jayson Tatum":"Luka Dončić"]

# %% [markdown]
# It follows that there are two ways of sorting a series: you can sort the labels of the index (and values have to be accordingly permuted to follow the ordering of the labels) or you can sort the values (and labels have to be accordingly  permuted to follow the ordering of the values):

# %%
points_by_player

# %%
points_by_player.sort_index()

# %%
points_by_player.sort_values()

# %% [markdown]
# Regardless of what the index is, selection of series points by position is always possible using `.iloc[]`:

# %%
points_by_player.iloc[0]

# %%
points_by_player.iloc[[0]]

# %%
points_by_player.iloc[0:2]

# %% [markdown]
# Series have many methods for doing basic calculations:

# %%
(points_by_player.min(), points_by_player.max())

# %%
points_by_player.mean()

# %%
points_by_player.sum()

# %% [markdown]
# You can also apply binary operators like `+`, `-`, `*`, `/` to a constant and a series resulting in a series where each element is the result of applying the binary operator to the corresponding element of the input series and the constant. For example, let us express the number of points as a fraction of the points scored by the player who scored the most in this season - we divide the series by a constant (the top score given by `points_by_player.iloc[0]`):

# %%
points_by_player / points_by_player.iloc[0]

# %% [markdown]
# We can also compute the fraction each player scored of the total the top five players scored:

# %%
points_by_player / points_by_player.sum()

# %% [markdown]
# ### Dataframes

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
# `iloc[]` returns rows and columns specified using list-like indexes (positive or negative offset of the row in the series or of the rows and optionally columns of the data frame):

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

# %%
df.groupby("cyl").mean(numeric_only=True)

# %% [markdown]
# You can also select a single series from the dataframe groupby object which results in a `SeriesGroupBy` object:

# %%
df.groupby("cyl")["mpg"]

# %% [markdown]
# Aggregations then produce a single value per group and result in a `pd.Series`:

# %%
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
# `agg(func)` calls `func(series)` once for each `series` of every group. `func` can be a function name, a function or a lambda expression and it should return a scalar value. The result of `agg` is a series or a dataframe depending whether aggregation is done on a single series or on a dataframe but also whether one aggregation is done or many. There are also multiple ways of providing arguments specifying the aggregations to do. Hence there are many cases which we now try to outline.

# %% [markdown]
# #### Aggregating a single series

# %% [markdown]
# The simplest aggregation is done on a single series and results in a series:

# %%
df = cars()
df.groupby("cyl")["hp"].agg("mean")

# %% [markdown]
# When you pass a list as the argument to `agg` the result will be a dataframe, since then in general there can be more than one group and more than one aggregation - the groups become rows and the aggregations become columns of the result:

# %%
df = cars()
df.groupby("cyl")["hp"].agg(["size", "mean", "std"])

# %% [markdown]
# This might be useful also with only one aggregation in the list, just to force the aggregation result to be a dataframe:

# %%
df = cars()
df.groupby("cyl")["hp"].agg(["mean"])

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
