import requests
import pandas as pd
import numpy as np
import os

DATA_URL = "https://urlab.be/api/space/openings/"


class BaseModel:
    def __init__(self):
        raise NotImplementedError("Implement init please")

    def get_cache_dir(self):
        return os.path.join("models", self.__class__.__name__)

    def get_past(self):
        next_url = DATA_URL
        data = []
        while next_url is not None:
            part = requests.get(next_url).json()
            next_url = part['next']
            data.extend(part['results'])

        df = pd.DataFrame(data)
        df.time = pd.to_datetime(df.time)
        df = df.sort('time')
        df = df.drop_duplicates('time')

        return df

    def get_past_resampled(self, freq='H'):
        df = self.get_past()

        start = df.time.iloc[0]
        end = df.time.iloc[-1]

        start = start.replace(minute=0, second=0, microsecond=0)
        end = end.replace(minute=0, second=0, microsecond=0)

        # Reindex on a monotonic time index
        index = pd.date_range(start=start, end=end, freq=freq)
        df = df.set_index('time').reindex(index, method='ffill')
        nans = np.isnan(df.is_open.astype(np.float64))
        df[nans] = not df[~nans].is_open[0]

        return df

    def train(self):
        raise NotImplementedError("Please implement train")

    def predict(self, past, lenght):
        raise NotImplementedError("Please implement predict")
