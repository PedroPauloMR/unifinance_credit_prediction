import os
import sys

import pandas as pd
import structlog
from sklearn.pipeline import Pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from utils.utils import load_config_file

logger = structlog.getLogger()


class DataPreprocess:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe
        self.trained_pipe = None

    def train(self, dataframe: pd.DataFrame):
        logger.info("Starting preprocessing...")
        self.trained_pipe = self.pipe.fit(dataframe)

    def transform(self, dataframe: pd.DataFrame):
        if self.trained_pipe is None:
            raise ValueError("Pipeline was not trained")
        logger.info("Initiating preprocessor data transformation...")
        data_processed = self.trained_pipe.transform(dataframe)
        return data_processed
