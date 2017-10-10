from base import BaseModel
import numpy as np
import os
import json


class Yolo(BaseModel):
    def __init__(self):
        with open(os.path.join(self.get_cache_dir(), "data.json"), 'w') as fd:
            self.MAGIC = json.reads(fd.read())["random_number"]

    def train(self):
        with open(os.path.join(self.get_cache_dir(), "data.json"), 'w') as fd:
            fd.write(json.dumps({"random_number": 4}))

    def predict(self, past, lenght):
        return np.random.rand(lenght) * self.MAGIC
