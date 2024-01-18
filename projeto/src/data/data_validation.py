import os
import sys
import pandas as pd

import pandera
from pandera import Check, Column, DataFrameSchema
import structlog

sys.path.append(os.path.join(os.path.dirname(__file__),"../src"))
from utils.utils import load_config_file

logger = structlog.getLogger()


class DataValidation:
    def __init__(self) -> None:
        self.columns_to_use = load_config_file().get('columns_to_use')

    def check_shape_data(self, dataframe: pd.DataFrame) -> bool:
        try:
            logger.info('Initiating validation...')
            dataframe.columns = self.columns_to_use
            return True
        except Exception as e:
            logger.error(f'Error on validation: {e}')
            return False
        
    def chek_columns(self, dataframe: pd.DataFrame) -> bool:
        schema = DataFrameSchema(
            {
                "target": Column(int, Check.isin([0, 1]), Check(lambda x: x > 0), coerce=True),
                "TaxaDeUtilizacaoDeLinhasNaoGarantidas": Column(float, nullable=True),
                "Idade": Column(int, nullable=True),
                "NumeroDeVezes30-59DiasAtrasoNaoPior": Column(int, nullable=True),
                "TaxaDeEndividamento": Column(float, nullable=True),
                "RendaMensal": Column(float, nullable=True),
                "NumeroDeLinhasDeCreditoEEmprestimosAbertos": Column(int, nullable=True),
                "NumeroDeVezes90DiasAtraso": Column(int, nullable=True),
                "NumeroDeEmprestimosOuLinhasImobiliarias": Column(int, nullable=True),
                "NumeroDeVezes60-89DiasAtrasoNaoPior": Column(int, nullable=True),
                "NumeroDeDependentes": Column(float, nullable=True)
            }
        )
        try:
            schema.validate(dataframe)
            logger.info('Validation columns passed...')
            return True
        except pandera.errors.SchemaErrors as exc:
            logger.error('Validation columns failed...')
            pandera.display(exc.failure_cases)
            return False
        
    def run(self, dataframe : pd.DataFrame) -> bool:
        if self.check_shape_data(dataframe) and self.chek_columns(dataframe):
            logger.info('Success on validate data')
            return True
        else:
            logger.error('Failed on validation')
            return False

