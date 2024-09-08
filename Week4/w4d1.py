import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme()
tips = sns.load_dataset("tips")
sns.relplot(
    data=tips,
    x="total_bill",
    y="tip",
    col="time",
    hue="smoker",
    style="smoker",
    size="size",
)

dots = sns.load_dataset("dots")
sns.relplot(
    data=dots,
    kind="line",
    x="time",
    y="firing_rate",
    col="align",
    hue="choice",
    style="choice",
    size="coherence",
    facet_kws=dict(sharex=False),
)

sns.lmplot(
    data=tips,
    x="total_bill",
    y="tip",
    col="time",
    hue="smoker",
)

# this is linear regression

# histogram
sns.displot(
    data=tips,
    x="total_bill",
    col="time",
)

# histogram
sns.displot(data=tips, x="total_bill", col="time", kde=True)

# cat plot is to cehck outliers
sns.catplot(
    data=tips,
    x="day",
    y="total_bill",
    kind="swarm",
    hue="smoker",
)


sns.catplot(
    data=tips,
    x="day",
    y="total_bill",
    kind="violin",
    hue="smoker",
)

sns.catplot(
    data=tips,
    x="day",
    y="total_bill",
    kind="bar",
    hue="smoker",
)

penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="flipper_length", y="bill_length", hue="species")


penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")

penguins.columns

# extensively use huncha yo
sns.pairplot(data=penguins, hue="species")
