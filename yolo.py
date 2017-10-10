from base import BaseModel
import numpy as np
import os
import json


class Yolo(BaseModel):
    def __init__(self):
        """
        Init is called once at the start of the server/thread and loads the model
        from it's cache directory (self.get_cache_dir())
        """
        with open(os.path.join(self.get_cache_dir(), "data.json"), 'w') as fd:
            self.MAGIC = json.reads(fd.read())["random_number"]

    def train(self):
        """
        Train will be called outside of the request/response cycle and has
        to save the model in it's cache dir (self.get_cache_dir())
        """
        with open(os.path.join(self.get_cache_dir(), "data.json"), 'w') as fd:
            fd.write(json.dumps({"random_number": 4}))

    def predict(self, past, lenght):
        """
        Predict is called at every query
        """
        return np.random.rand(lenght) * self.MAGIC
