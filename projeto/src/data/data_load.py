import os
import sys
import pandas as pd
import structlog
sys.path.append(os.path.join(os.path.dirname(__file__),"../src"))
from utils.utils import load_config_file

logger = structlog.getLogger()

class DataLoad:
    """Class data load"""
    def __init__(self) -> None:
        pass
    
    def load_data(self,dataset_name: str) -> pd.DataFrame:
        """This function will load the dataset from the filename passed
        Args:
        dataset_name (str): name of dataset

        return:
        pandas DataFrame"""

        logger.info(f'Initiating data load with name: {dataset_name}')

        dataset = load_config_file().get(dataset_name)
        try:
            dataset = load_config_file().get(dataset_name)
            if dataset is None:
                raise ValueError(f'Error: the name passed of dataset is incorrect: {dataset}')
            loaded_data = pd.read_csv(f'../data/raw/{dataset}')    
            return loaded_data[load_config_file().get('columns_to_use')]
        
        except ValueError as ve:
            logger.error(str(ve))
        
        except Exception as e:
            logger.error(f'Unexpected error: {str(e)}')
