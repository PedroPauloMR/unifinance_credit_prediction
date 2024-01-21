from mlflow.deployments import get_deploy_client


app_name = 'prob-loan-sagemaker'
arn = 'arn:aws:iam' # role sagemaker
image_ecr_uri = 'imagem:tag_id' # ecr
region = 'us-east-1'

model_uri = 'C:/repos/portfolio_projetos/unifinance_credit_prediction/projeto/mlartifacts/1/601eff740cce4faf9253315963aa980d/artifacts/modelo.joblib'

config = dict(
    execution_role_arn = arn,
    bucket_name='s3_bucket',
    image_url = image_ecr_uri,
    region_name = region,
    archive = False,
    instance_type = 'ml.m4.xlarge',
    instance_count = 1,
    synchronous = True,
    timeout_seconds = 3600,
    variant_name = 'prod-variant-3'
)

client = get_deploy_client('sagemaker')
deploy_client = client.create_deployment(app_name, model_uri=model_uri,flavor = 'python_function',config = config)
print(f'deploy_client: {deploy_client}')