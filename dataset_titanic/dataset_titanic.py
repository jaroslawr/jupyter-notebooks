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

# %%
pd.options.display.precision = 2

# %% [markdown]
# ## Read data

# %%
df = pd.read_csv("titanic.txt")
df.head(5)


# %% [markdown]
# ## Analysis

# %% [markdown]
# ### Number of passengers and survivors

# %%
def survivors(df):
    return len(df[df == 1])

def survivors_pct(df):
    if len(df) > 0:
        return len(df[df == 1]) / len(df)
    else:
        return 0.0


# %%
len(df)

# %%
survivors(df["survived"])

# %%
survivors_pct(df["survived"])

# %% [markdown]
# ### Survivors by gender

# %%
survivors_by_sex = df.groupby("sex")["survived"].agg(
    survivors_pct=survivors_pct,
    survivors=survivors,
    total=len
)
survivors_by_sex

# %%
sns.catplot(data=df, y="sex", hue="survived", kind="count")

# %%
sns.catplot(data=df, y="sex", x="survived", kind="point", join=False, orient="h", capsize=0.25)

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
sns.catplot(data=df, y="pclass", hue="survived", kind="count")

# %%
sns.catplot(data=df, y="pclass", x="survived", kind="point", join=False, orient="h", capsize=0.25)

# %% [markdown]
# ### Divide age into buckets

# %%
(df["age"].min(), df["age"].max())

# %%
bins = np.arange(0.0, 90.0, 10.0)
bins

# %%
df["age_bucket"] = pd.cut(df["age"], bins=bins)
df.head(3)

# %% [markdown]
# ### Survivors by age

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
# ### Survivors by gender and passenger class

# %%
survivors_by_class_and_sex = df.groupby(["pclass", "sex"])["survived"].agg(
    survivors_pct=survivors_pct,
    survivors=survivors, 
    total=len
)
survivors_by_class_and_sex.unstack(level=0)

# %%
sns.catplot(data=df, y="sex", hue="survived", col="pclass", kind="count", height=4)

# %%
sns.catplot(data=df, y="sex", x="survived", col="pclass", kind="point", join=False, orient="h", capsize=0.25, height=4)

# %% [markdown]
# ### Survivors by age and passenger class

# %%
survivors_by_class_and_age = df.groupby(["pclass", "age_bucket"])["survived"].agg(
    survivors_pct=survivors_pct,
    survivors=survivors, 
    total=len
)
survivors_by_class_and_age.unstack(level=0)

# %%
sns.catplot(data=df, y="age_bucket", hue="survived", col="pclass", kind="count", height=4)

# %%
sns.catplot(data=df, y="age_bucket", x="survived", col="pclass", kind="point", join=False, orient="h", capsize=0.25, height=4)
