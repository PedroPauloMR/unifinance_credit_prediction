import os
import yaml

def load_config_file():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_relativo = os.path.join('..','..','config','config.yaml')

    config_file_path = os.path.abspath( os.path.join(diretorio_atual,caminho_relativo) ) 

    config_file = yaml.safe_load(open(config_file_path, 'rb'))
    # print(config_file)
    return config_file


def path_model_trained():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_relativo = os.path.join('..','..','models')
    model_path = os.path.abspath( os.path.join(diretorio_atual,caminho_relativo) ) 
    return model_path



