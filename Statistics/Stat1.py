import numpy as np
import pandas as pd

epa_data = pd.read_csv("c4_epa_air_quality.csv", index_col=0)

epa_data.head(10)

epa_data.describe()

epa_data["state_name"].describe()

np.mean(epa_data["aqi"])

np.median(epa_data["aqi"])

np.min(epa_data["aqi"])

np.std(epa_data["aqi"], ddof=1)
