import boto3
import json
import os
from fastapi import FastAPI, Request, HTTPException


endpoint_name = os.environ.get("ENDPOINT_NAME")
region_name = os.environ.get("REGION_NAME")
assert endpoint_name is not None, "ENDPOINT_NAME environment variable is not set"
assert region_name is not None, "REGION_NAME environment variable is not set"
print(f"Endpoint name: {endpoint_name}")
print(f"Region name: {region_name}")

app = FastAPI()

sm = boto3.client("sagemaker", region_name=region_name)
sm_rt = boto3.client("sagemaker-runtime", region_name=region_name)


@app.get("/")
def ping():
    ''' Return ping message'''
    return {"endpoint name": endpoint_name, "region name": region_name}

@app.get("/list_endpoints")
def list_endpoints():
    ''' List all SageMaker endpoints'''
    endpoints = sm.list_endpoints()
    endpoint_details = []
    for endpoint in endpoints['Endpoints']:
        endpoint_description = sm.describe_endpoint(EndpointName=endpoint['EndpointName'])
        endpoint_config = sm.describe_endpoint_config(EndpointConfigName=endpoint_description['EndpointConfigName'])
        production_variant = endpoint_config['ProductionVariants'][0]
        
        model_name = production_variant['ModelName']
        model_details = sm.describe_model(ModelName=model_name)
        
        endpoint_details.append({
            'EndpointName': endpoint['EndpointName'],
            'InstanceType': production_variant['InstanceType'],
            'Container': model_details['PrimaryContainer']['Image'],
            'ModelEnvironment': model_details['PrimaryContainer'].get('Environment', {}),
        
        })
        
    
    return endpoint_details

@app.post("/predict")
async def predict(request: Request):
    ''' Invoke the SageMaker endpoint'''
    try:
        payload = await request.json()
        response = sm_rt.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        result = json.loads(response['Body'].read().decode())
        return result
    except json.JSONDecodeError as json_err:
        raise HTTPException(status_code=500, detail=f"Invalid JSON input: {str(json_err)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")