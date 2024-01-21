import json

import pandas as pd 
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

workspace_name = ''
workspace_location = 'EastUS'
resource_group = 'res_group'
subscription_id = 'subs_id'
endpoint_name = 'ple'

ml_client = MLClient(DefaultAzureCredential(),subscription_id, resource_group, workspace_name)
# print(ml_client)

df_test = pd.read_csv(r'C:\repos\portfolio_projetos\unifinance_credit_prediction\projeto\data\raw\test.csv')
data = {"input_data":df_test.iloc[[0]].to_dict(orient = 'split')}

print('\n')
print(data)
print('\n')

with open('file.json','w') as f:
    json.dump(data, f)

response = ml_client.online_endpoints.invoke(endpoint_name = endpoint_name, request_file = 'file.json')

print(response)