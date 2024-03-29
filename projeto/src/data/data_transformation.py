import os
import sys

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from utils.utils import load_config_file

logger = structlog.getLogger()


class DataTransformation:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.target_name = load_config_file().get("target_name")

    def train_test_spliting(self):
        X = self.dataframe.drop(self.target_name, axis=1)
        y = self.dataframe[self.target_name]

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=load_config_file().get("test_size"),
            stratify=y,
            random_state=load_config_file().get("random_state"),
        )
        return X_train, X_val, y_train, y_val
