import json
import boto3
import pandas as pd 

app_name = 'prob-loan-sagemaker'
region = 'us-east-1'

df_test = pd.read_csv(r'C:\repos\portfolio_projetos\unifinance_credit_prediction\projeto\data\raw\test.csv')

def query(input_json):
    client = boto3.session.Session().client('sagemaker-runtime',region)
    response = client.invoke_endpoint(EndpointName = app_name, Body = input_json,
                                      ContentType = 'application/json')
    preds = response['Body'].read().decode('ascii')
    preds = json.loads(preds)
    print(f'Received answer: {preds}')
    return preds


# manipulation
data = {"dataframe_split":df_test.iloc[[0]].to_dict(orient = 'split')}
byte_data = json.dumps(data).encode('utf-8')

output = query(byte_data)

resp = pd.DataFrame([output])
print(resp)
