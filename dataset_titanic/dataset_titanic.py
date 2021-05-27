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
# # Data analysis: Survivors of sinking of the Titanic

# %% [markdown]
# Data analysis of survivors of sinking of the Titanic, using data from:
#
# <https://hbiostat.org/data/>

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
plt.style.use("ggplot")

# %% [markdown]
# ## Read data

# %%
df = pd.read_csv("titanic.txt")
df.head(5)


# %% [markdown]
# ## Analysis

# %%
def survivors(df):
    return len(df[df == 1])

def survivors_pct(df):
    return len(df[df == 1]) / len(df)


# %% [markdown]
# ### Survivors by age

# %%
df["age_bucket"] = np.floor(df["age"] / 10.0)
df.head(3)

# %%
survivors_by_age = df.groupby("age_bucket")["survived"].agg(
    survivors_pct=survivors_pct,
    survivors=survivors,
    total=len
)
survivors_by_age

# %%
sns.catplot(data=df, y="age_bucket", hue="survived", kind="count")

# %%
sns.catplot(data=df, y="age_bucket", x="survived", kind="point", join=False, orient="h", capsize=0.25)

# %% [markdown]
# ### Survivors by passenger class

# %%
survivors_by_class = df.groupby("pclass")["survived"].agg(
    survivors_pct=survivors_pct,
    survivors=survivors, 
    total=len
)
survivors_by_class

# %%
sns.catplot(data=df, y="age_bucket", hue="survived", kind="count")

# %%
sns.catplot(data=df, y="pclass", x="survived", kind="point", join=False, orient="h", capsize=0.25)

# %% [markdown]
# ### Survivors by passenger class and age

# %%
survivors_by_class_and_age = df.groupby(["pclass", "age_bucket"])["survived"].agg(
    survivors_pct=survivors_pct,
    survivors=survivors, 
    total=len
)
survivors_by_class_and_age

# %%
sns.catplot(data=df, y="age_bucket", hue="survived", col="pclass", kind="count", height=4)

# %%
sns.catplot(data=df, y="age_bucket", x="survived", col="pclass", kind="point", join=False, orient="h", capsize=0.25, height=4)
