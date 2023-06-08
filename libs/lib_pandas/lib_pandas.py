# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data analysis with Pandas

# %% [markdown]
# This is a guide to doing data analysis with Pandas written with a particular goal: to help understand the most important concepts and usage patterns in a way that sticks with you, as opposed to covering every possible application. We try to achieve this by laying out and following some specific principles:
#
# - Build deep understanding of general concepts:
#     - the basic Panas data structures: series, dataframes and indexes
#     - wide classes of operations on data: selection, grouping, aggregations and transformations
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
# ## Pandas data structures

# %% [markdown]
# We begin by looking into the fundamental high-level Pandas data structures: series, data frames and indexes. If you try to use Pandas without understanding the data structures and the relations between them, it gets confusing very easily. On the other hand learning the data structures can feel dry compared to working with real datasets. We cover only the most important basics here to keep things interesting and later revisit in more depth when need arises.

# %% [markdown]
# ### Series and basic indexes

# %% [markdown]
# A series holds an array of values:

# %%
primes = pd.Series([2, 3, 5, 7, 11])
primes

# %% [markdown]
# The right column of the output simply shows the values in the series. `dtype: int64` refers to the data type of the values.

# %% [markdown]
# The left column of the output shows labels corresponding to the values in the series. The labels are part of the *index* that is a part of every series. By default the series is indexed simply using position of each value in the series:

# %%
primes = pd.Series([2, 3, 5, 7, 11])
primes.index

# %% [markdown]
# `.loc[]` call for a series takes an index label as argument and looks up the corresponding value in the series. For the default index it ends up working like basic array indexing:

# %%
primes = pd.Series([2, 3, 5, 7, 11])
primes.loc[0]

# %% [markdown]
# The index is what makes a series something more than a simple array. It makes it possible to refer to the values in the series by whatever label is appropriate: by a string, by a date, by a pair of numbers, ... It also plays a role similar to a database index, hence the name: it speeds up operations like joins that have to lookup by this label repeatedly. Here is how you construct a series with an explicit index that does not simply use position in the series as the label of each value:

# %%
primes = pd.Series([2, 3, 5, 7, 11], index=["p0", "p1", "p2", "p3", "p4"])
primes

# %% [markdown]
# Now you can lookup by label:

# %%
primes = pd.Series([2, 3, 5, 7, 11], index=["p0", "p1", "p2", "p3", "p4"])
primes.loc["p0"]

# %% [markdown]
# Lookup by index is nevertheless always possible using `.iloc[]`:

# %%
primes = pd.Series([2, 3, 5, 7, 11], index=["p0", "p1", "p2", "p3", "p4"])
primes.iloc[2]

# %% [markdown]
# You can explicitly construct a `pd.Index` instance and pass it in the `pd.Series` constructor:

# %%
index = pd.Index(["p0", "p1", "p2", "p3", "p4"])
series = pd.Series([2, 3, 5, 7, 11], index=index)
series

# %% [markdown]
# ### Data frames

# %% [markdown]
# Conceptually a dataframe is a collection of column series that share a common row index:

# %%
df = pd.DataFrame({"a": series, "b": series})
df

# %%
df = pd.DataFrame({"a": series, "b": series})
display(id(df.index))
display(id(df["a"].index))
display(id(df["b"].index))

# %% [markdown]
# Columns get labels through an additional index object, called the column index:

# %%
df = pd.DataFrame({"a": series, "b": series})
df.columns

# %% [markdown]
# Many Pandas methods work with either the row index or column index depending on the value of the `axis` keyword argument. For example removing rows by label and removing columns using a label or a set of labels are both done using the `drop` method:

# %%
df = pd.DataFrame({"a": series, "b": series})
df.drop(["p0", "p1"])

# %%
df = pd.DataFrame({"a": series, "b": series})
df.drop("a", axis=1)

# %% [markdown]
# ### Jupyter representation of the data structures

# %% [markdown]
# It is good to learn how the Jupyter cell output corresponds to the underlying Pandas data structures and its attributes. Often we will not construct a `pd.Series` or `pd.DataFrame` directly, but receive one as a result of some sequence of Pandas operations. In this case we do not know up front whether it is a `pd.Series` or `pd.DataFrame`, what the index is, whether the index has a name, etc. You can inspect the result, for example call `type(result)`, but it is more efficient to simply learn what is shown where and how.

# %% [markdown]
# A `pd.Series` is displayed in monospaced font (unlike `pd.DataFrame`) and as the below example illustrates its Jupyter representation has at most four parts:
#
# - top left corner: index name, only shown if present
# - left column: index labels  
# - right column: series values
# - bottom line: series name, if present, and dtype of series values

# %%
index = pd.Index(["p0", "p1", "p2", "p3", "p4"], name="key")
series = pd.Series([2, 3, 5, 7, 11], index=index, name="primes")
series

# %% [markdown]
# A dataframe with one column is something different than a series:

# %%
index = pd.Index(["p0", "p1", "p2", "p3", "p4"], name="key")
series = pd.Series([2, 3, 5, 7, 11], index=index, name="primes")
series.to_frame()

# %% [markdown]
# ## Dataset: mtcars

# %% [markdown]
# We will use the mtcars dataset as the default basic example in the following sections. Its contents are easy to understand and fairly interesting, while it also is provided in one of the simplest and most commonly encountered ways: each row describes a single unique unit of observation (a car) and each column is a different measurement for the same unit of observation (miles/gallon, number of cylinders, horsepower etc.):

# %%
CARS = pd.read_csv("mtcars.csv")

def cars():
    # Return a fresh copy every time so that examples do not affect each other
    return CARS.copy()


# %% [markdown]
# ## Examining datasets

# %% [markdown]
# `.head(n)` returns the first `n` rows, `.tail(n)` the last `n` rows, and `.sample(n)` random `n` rows:

# %%
df = cars()
display(df.head(2))
display(df.tail(2))
display(df.sample(2))

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
# Describe will also show the number of non-null rows for given column, but along with basic summary statistics:

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
# ## Selecting rows and columns

# %% [markdown]
# ### Select with []

# %% [markdown]
# Select rows with `[]`:

# %%
df = cars()
df[df["gear"] == 5]

# %% [markdown]
# `[]` will accept a callable as argument for cases where a reference to the dataframe is not available, for example when chaining method calls on the dataframe:

# %%
df = cars()
df[lambda df: df["gear"] == 5][lambda df: df["cyl"] == 4]

# %% [markdown]
# Select a single column as a `pd.Series` with `[]`:

# %%
df = cars()
df["gear"].head(5)

# %% [markdown]
# Select one or more columns as a `pd.DataFrame` by passing a list to `[]`:

# %%
df = cars()
df[["gear"]].head(5)

# %% [markdown]
# Selection of rows and of columns can be combined:

# %%
df = cars()
df[df["gear"] == 5]["cyl"]

# %% [markdown]
# Note that chaining `[]` does not work for the purpose of modifying or inserting data:

# %%
df = cars()
df[df["gear"] == 5]["cyl"] = 3

# %% [markdown]
# `df[][]=` translates to a `df.__getitem__()` call on the data frame and then a `.__setitem__()` call on the resulting object. The problem is that the `df.__getitem__()` call might return either a view or a copy of the dataframe, so the dataframe might or might not be modified.
#
# Instead, `df.loc[]` can be used to select rows and columns at the same time. `df.loc[]` will return a view or a copy just like `df[]`, but `df.loc[]=` is just a single method call on the `loc` attribute of the original dataframe, free of the ambiguity of `[][]=`, so that it will always correctly modify the dataframe.

# %% [markdown]
# ### Select with loc[]

# %% [markdown]
# Select rows:

# %%
df = cars()
df.loc[df["gear"] == 5]

# %% [markdown]
# `loc[]` will accept a callable as argument for cases where a reference to the dataframe is not available, for example when chaining method calls on the dataframe:

# %%
df = cars()
df.loc[lambda df: df["gear"] == 5].loc[lambda df: df["cyl"] == 4]

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
# ## Grouping

# %% [markdown]
# ### Forming groups

# %% [markdown]
# ### Aggregating group data

# %% [markdown]
# #### Aggregate with specific provided aggregations

# %% [markdown]
# `df.groupby()` returns a `DataFrameGroupBy` that supports calls like `min()`, `max()`, `mean()`, `std()`, `var()`, `quantile()` etc.:

# %%
df = cars()
df.groupby(["cyl", "gear"])["hp"].mean()

# %% [markdown]
# To count the number of items in each group use `.size()`:

# %%
df = cars()
df.groupby(["cyl", "gear"]).size()

# %% [markdown]
# Note that confusingly to get the number of rows in the whole dataframe, you have to do `len(df)` or `len(df.index)` - in a dataframe `.size` is a field not a method and it holds the number of cells in the dataframe not the number of rows. For example, to group cars by the number of cylinders and number of gears and see what percentage of all cars have what setup you call `.size()` on the `DataFrameGroupBy` object returned by `groupby(["cyl", "gear"])`, divide by `len(df)` and multiply the result by `100.0`:

# %%
df = cars()
(df.groupby(["cyl", "gear"]).size() / len(df)) * 100.0

# %% [markdown]
# To count the groups themselves use the `.ngroups` attribute:

# %%
df = cars()
df.groupby(["cyl", "gear"]).ngroups

# %% [markdown]
# To compute a compound expression involving group level aggregates, for example to compute the range of values within each group (group max - group min), reference to the groupby object and reuse it:

# %%
df = cars()
df_groupby = df.groupby(["cyl", "gear"])["hp"]
df_groupby.max() - df_groupby.min()

# %% [markdown]
# `.pipe()` will accomplish the same thing in a way that can be placed in the middle of a method chain:

# %%
(
    cars()
    .groupby(["cyl", "gear"])["hp"]
    .pipe(lambda group: group.max() - group.min())
    .mean()
)

# %% [markdown]
# #### Aggregate with generic aggregate() method (series-by-series)

# %% [markdown]
# `df.groupby().aggregate(func)` will call `func(series)` once for each series of every group.
#
# `func` should return a scalar.

# %%
df = cars()
df.groupby("cyl")[["hp"]].aggregate("mean")

# %% [markdown]
# Multiple aggregations can be specified:

# %%
df = cars()
df.groupby("cyl")[["hp"]].aggregate(["size", "mean", "std"])

# %% [markdown]
# Use keyword arguments to rename the resulting columns:

# %%
df = cars()
df.groupby("cyl").aggregate(average=("hp", "mean"))

# %% [markdown]
# The last type of aggregate() aggregation has slightly different syntax when dealing with a single series:

# %%
df = cars()
df.groupby("cyl")["hp"].agg(average=np.mean)

# %% [markdown]
# #### Aggregate with generic apply() method (dataframe-by-dataframe)

# %% [markdown]
# `df.groupby().apply(func)` will call `func(group)` once for each group, where `group` is a dataframe containing the rows within each group. The form of the result depends of the return type of `func`.

# %% [markdown]
# **Case 1:** `func` returns a scalar - the result of `apply(func)` is a series indexed by the group key:

# %%
df = cars()
df.groupby("cyl")[["hp"]].apply(lambda df: np.mean(df.values))

# %% [markdown]
# **Case 2:** `func` returns a series - the result of `apply(func)` is a dataframe indexed by the group key, with columns given by the index of the returned series:

# %%
df = cars()
df.groupby("cyl")[["hp", "mpg"]].apply(lambda df: df.mean())

# %% [markdown]
# **Case 3:** `func` returns a dataframe - the result of `apply(func)` is a dataframe with a multi-index and with same columns as the dataframe returned by `func`. The multi-index consists of a group key level concatenated with levels of the index of the dataframes returned by `func`:

# %%
df = cars()
df.groupby("cyl").apply(lambda df: df.drop(columns=["Currency"]).set_index(["Year"]).rolling(3).mean().dropna())

# %% [markdown]
# ### Transforming group data

# %% [markdown]
# In Pandas terminology, transformations differ from aggregations in that the result of a transformation has dimensions equal to the dimensions of the original series or dataframe before the `.groupby()`. Common use case is to do something with each row of a dataframe but using some data computed in the context of the group of given row, like the group mean of some attribute.

# %% [markdown]
# #### Transform with specific provided transformations

# %% [markdown]
# #### Transform rows one-by-one with generic transform() method

# %% [markdown]
# `df.groupby().transform(func)` will call `func(series_in_group)` once for each series in each group. In contrast to `apply()`, the result of `transform()` is of the same dimensions as the original dataframe.
#
# `func(series_in_group)` should either return a series of the same dimensions as `series_in_group` or a scalar, in which case pandas will take care of making a series of length `len(series_in_group)` out of it.

# %%
df = usd_exchange_rates_df()
df.groupby("Currency")[["Currency/USD", "USD/Currency"]].transform(lambda df: df.mean())
