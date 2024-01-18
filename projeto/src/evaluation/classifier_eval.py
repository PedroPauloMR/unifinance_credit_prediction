import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

import structlog
sys.path.append(os.path.join(os.path.dirname(__file__),"../src"))
from utils.utils import load_config_file,path_model_trained

logger = structlog.getLogger()

class ModelEvaluation:
    def __init__(self, model, X, y, n_splits = 5):
        self.model = model
        self.X = X
        self.y = y
        self.n_splits = n_splits 
    
    def cross_val_evaluate(self):
        logger.info('Initiating cross validation...')
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=load_config_file().get('random_state'))
        scores = cross_val_score(self.model, self.X, self.y,cv = skf,scoring='roc_auc')
        return scores
    
    def roc_auc_scorer(self, model, X, y):
        y_pred = model.predict_proba(X)[:,1]
        return roc_auc_score(y, y_pred)
    
    @staticmethod
    def evaluate_predictions(y_true, y_pred_proba):
        logger.info('Initiating model validation...')
        return roc_auc_score(y_true, y_pred_proba)