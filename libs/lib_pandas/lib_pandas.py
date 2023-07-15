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
# `.loc[]` call for a series takes an index label as argument and looks up the corresponding value in the series. For the default index it ends up working like basic array indexing (though it does not support negative indexing, since it does label lookup):

# %%
primes = pd.Series([2, 3, 5, 7, 11])
primes.loc[2]

# %% [markdown]
# The index is what makes a series something more than a simple array. It makes it possible to refer to the values in the series by whatever label is appropriate: by a string, by a date, by a pair of numbers, ... It also plays a role similar to a database index, hence the name: it speeds up operations like joins that have to lookup by label repeatedly. Here is how you construct a series with an explicit index that does not simply use position in the series as the label of each value:

# %%
primes = pd.Series([2, 3, 5, 7, 11], index=["p0", "p1", "p2", "p3", "p4"])
primes

# %% [markdown]
# Now you can lookup values by label or a set of labels:

# %%
primes = pd.Series([2, 3, 5, 7, 11], index=["p0", "p1", "p2", "p3", "p4"])
primes.loc["p0"]

# %%
primes = pd.Series([2, 3, 5, 7, 11], index=["p0", "p1", "p2", "p3", "p4"])
primes.loc[["p0", "p2"]]

# %% [markdown]
# ### Data frames

# %% [markdown]
# Conceptually a dataframe is a collection of column series that share a common row index:

# %%
df = pd.DataFrame({"a": primes, "b": primes})
df

# %%
df = pd.DataFrame({"a": primes, "b": primes})
display(id(df.index))
display(id(df["a"].index))
display(id(df["b"].index))

# %% [markdown]
# Columns get labels through an additional index object, called the column index:

# %%
df = pd.DataFrame({"a": primes, "b": primes})
df.columns

# %% [markdown]
# Many Pandas methods work with either the row index or column index depending on the value of the `axis` keyword argument. For example removing rows by label and removing columns using a label or a set of labels are both done using the `drop` method:

# %%
df = pd.DataFrame({"a": primes, "b": primes})
df.drop(["p0", "p1"])

# %%
df = pd.DataFrame({"a": primes, "b": primes})
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
# ## Grouping

# %% [markdown]
# ### Splitting into groups

# %%

# %% [markdown]
# ### Aggregating using predefined methods

# %% [markdown]
# The series groupby and dataframe groupby objects both support calls like `min()`, `max()`, `mean()`, `std()`, `var()`, `quantile()` etc. that are groupwise versions of the respective series/dataframe operation:

# %%
df = cars()
df.groupby(["cyl", "gear"])["hp"].mean()

# %% [markdown]
# To count the number of rows in each group call `.size()` on the groupby object:

# %%
df = cars()
df.groupby(["cyl", "gear"]).size()

# %% [markdown]
# Confusingly, to get the number of rows in the whole dataframe, you need to call `len(df)` or `len(df.index)` - `.size` in a dataframe is a field not a method and it holds the number of cells in the dataframe not the number of rows. For example, to group cars by the number of cylinders and number of gears and see what percentage of all cars have what setup you call `.size()` on the groupby object returned by `groupby(["cyl", "gear"])`, divide by `len(df)` and multiply the result of the division by `100.0`:

# %%
df = cars()
(df.groupby(["cyl", "gear"]).size() / len(df)) * 100.0

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
# `agg(func)` calls `func(series)` once for each `series` of every group. `func` should return a scalar. `func` can be passed in as a function or a function name (a string). The result is a series or a dataframe depending whether aggregation is done on a single series or on a dataframe but also whether one aggregation is done or more. There are also multiple ways of providing arguments specifying the aggregations to do. Hence there are many cases which we now try to showcase.

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
