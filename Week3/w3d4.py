import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
df = pd.DataFrame(
    {
        "key": ["A", "B", "C", "A", "B", "C"],
        "data1": range(6),
        "data2": rng.randint(0, 10, 6),
    },
    columns=["key", "data1", "data2"],
)

df

# reserved keywords ma chai = halni ani hami le define gareko ma chai colon

df.groupby("key").aggregate([min, np.mean, np.max])
# list ma pass gareko

# aba chai dataframe ma
df.groupby("key").aggregate({"data1": "min", "data2": "max"})

# transformation
df.groupby("key").transform(lambda x: x - x.mean())

from dateutil import parser  # noqa: E402

date = parser.parse("3rd of may, 2016")
date

from datetime import datetime  # noqa: E402

datetime(year=2016, month=7, day=24)

# both of them are native python functions

date.strftime("%A")

# tyo timee ma k day pareko cha?

date = pd.to_datetime("5th of august, 1963")

date

date.strftime("%A")

date + pd.to_timedelta(np.arange(12), "H")

index = pd.DatetimeIndex(pd.date_range("2015-1-22", "2020-1-22", periods=4))
data = pd.Series([0, 1, 2, 3], index=index)
data

pd.period_range("2016-07", periods=7, freq="M")

msft = pd.read_csv("MSFT_Stock_data.csv")
msft.tail()

ls  # type: ignore # noqa: F821

msft.set_index("Date", inplace=True)
msft.head()

# set index is explicit always

# check notes here
