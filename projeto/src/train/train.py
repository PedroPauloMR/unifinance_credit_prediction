import os
import sys
import pandas as pd
import joblib

import structlog
sys.path.append(os.path.join(os.path.dirname(__file__),"../src"))
from utils.utils import load_config_file,path_model_trained

logger = structlog.getLogger()

class TrainModels:
    def __init__(self, dados_X: pd.DataFrame,dados_y: pd.DataFrame):
        self.dados_X = dados_X 
        self.dados_y = dados_y
        self.model_name = load_config_file().get('model_name')
        self.model = None
        
    def train(self, model):
        model.fit(self.dados_X, self.dados_y)
        self.model = model
        return model
    
    def save_model(self):
        path = path_model_trained()
        joblib.dump(self.model, path + '/' + self.model_name)