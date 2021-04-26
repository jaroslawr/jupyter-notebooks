# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data analysis: Iris dataset

# %% [markdown]
# Dataset from:
#
# https://archive.ics.uci.edu/ml/datasets/iris

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Setup

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## Read CSV

# %%
df = pd.read_csv("iris.data", names=("sepal_length", "sepal_width", "petal_length", "petal_width", "class"))
df.head(3)

# %% [markdown]
# ## Examine

# %%
df[["class"]].value_counts()


# %% [markdown]
# ## Make plots

# %% [markdown]
# ### Run sequence plot

# %%
def seqplot(attr, w, h):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.set_size_inches(w, h)

    for i, (species, species_df) in enumerate(df.groupby("class")):
        axs[i].plot(np.linspace(0, len(species_df), len(species_df)), species_df[attr], "o", fillstyle="none")
        axs[i].set_title(species)
        
seqplot(attr="sepal_length", w=16, h=3)

# %% [markdown]
# ### Strip plot

# %%
sns.catplot(data=df, x="sepal_length", y="class", kind="strip", height=4, aspect=2, alpha=0.5, s=10)

# %% [markdown]
# ### Histogram

# %%
sns.displot(df, x="sepal_length", hue="class", stat="probability", element="step", 
            bins=15, common_norm=False, height=4, aspect=2, alpha=0.5, edgecolor="black")

# %% [markdown]
# ### ECDF

# %%
sns.displot(data=df, x="sepal_length", hue="class", kind="ecdf", height=4, aspect=2)

# %% [markdown]
# ### KDE

# %%
sns.displot(data=df, x="sepal_length", hue="class", kind="kde", height=4, aspect=2, fill=True, alpha=0.5, linewidth=2)

# %% [markdown]
# ### Box plot

# %%
sns.catplot(data=df, x="sepal_length", y="class", kind="box", height=4, aspect=2)
