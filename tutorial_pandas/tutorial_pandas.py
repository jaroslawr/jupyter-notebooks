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
# ## Imports

# %%
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## Settings

# %% [markdown]
# ### Maximum number of DF rows and columns to show

# %%
pd.options.display.max_rows = 999
pd.options.display.max_columns = 100

# %% [markdown]
# ### Plot style

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## Basic concepts

# %% [markdown]
# ## Reading data

# %% [markdown]
# ## Inspecting data

# %% [markdown]
# ## Cleaning data

# %% [markdown]
# ### Long format to wide

# %% [markdown]
# ### Wide format to long

# %% [markdown]
# ## Transforming data

# %% [markdown]
# ## Aggregating data

# %% [markdown]
# ## Plotting data

# %% [markdown]
# Pandas plotting methods expect data to be in the wide format, with different series in different columns.

# %% [markdown]
# ## Exporting data
