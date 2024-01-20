import os
import sys
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from feature_engine.imputation import MeanMedianImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression

import structlog
sys.path.append(os.path.join(os.path.dirname(__file__),"../src"))
from utils.utils import load_config_file,path_model_trained
from evaluation.classifier_eval import ModelEvaluation

import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('prob_loan')

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
    
    def get_best_model(self):
        logger.info('Obtaining the best model from MLFLOW...')
        df_mlflow = mlflow.search_runs(filter_string='metrics.val_roc_auc < 1').sort_values('metrics.val_roc_auc',ascending = False)
        run_id = df_mlflow.loc[df_mlflow['metrics.val_roc_auc'].idxmax()]['run_id']
        df_best_params = df_mlflow.loc[df_mlflow['run_id'] == run_id][['params.fit_intercept', 'params.solver',
                        'params.C', 'params.scaler', 'params.class_weight',
                        'params.discretizer', 'params.multi_class', 'params.max_iter',
                        'params.imputer', 'params.tol', 'params.warm_start']]
        best_roc_auc = df_mlflow.loc[df_mlflow['metrics.val_roc_auc'].idxmax()]['metrics.val_roc_auc']
        return df_best_params, best_roc_auc
    
    def run(self):
        df_best_params, _ = self.get_best_model()
        logger.info(f'Starting model training: {self.model_name}')
        with mlflow.start_run(run_name = 'final_model'):
            mlflow.set_tag('model_name',self.model_name)

            model = LogisticRegression(
                warm_start = eval(df_best_params["params.warm_start"].values[0]),
                multi_class = df_best_params["params.multi_class"].values[0],
                class_weight = df_best_params["params.class_weight"].values[0],
                max_iter = int(df_best_params["params.max_iter"].values[0]),
                C = float(df_best_params["params.C"].values[0]),
                solver = df_best_params["params.solver"].values[0],
                tol = float(df_best_params["params.tol"].values[0]),
            )

            pipe = Pipeline(
                [
                    ("imputer", eval(df_best_params["params.imputer"].values[0])),
                    (
                        "discretizer",
                        eval(df_best_params["params.discretizer"].values[0]),
                    ),
                    ("scaler", eval(df_best_params["params.scaler"].values[0])),
                    ("model", model),
                ]
            )
            pipe.fit(self.dados_X, self.dados_y)

            # logar metricas a de avaliacao
            y_preds = pipe.predict_proba(self.dados_X)[:, 1]
            model_eval = ModelEvaluation(model, self.dados_X, self.dados_y)
            val_roc_auc = model_eval.evaluate_predictions(self.dados_y, y_preds)
            mlflow.log_metric("valid_roc_auc", val_roc_auc)

            # registrar o modelo
            mlflow.sklearn.log_model(
                pipe,
                self.model_name,
                pyfunc_predict_fn="predict_proba",
                input_example=self.dados_X.iloc[[0]],
                registered_model_name=self.model_name,
            )

    def save_model(self):
        path = path_model_trained()
        joblib.dump(self.model, path + '/' + self.model_name)