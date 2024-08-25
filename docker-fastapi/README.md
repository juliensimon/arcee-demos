# SageMaker Endpoint FastAPI Wrapper

This project provides a Docker-based FastAPI wrapper for a SageMaker endpoint, with optional CloudFormation deployment.

## Description

This application deploys an Arcee model to Amazon SageMaker and creates a FastAPI server that acts as a wrapper for the SageMaker endpoint. It enables users to make predictions using the deployed model through a simple API interface.

## Setup Instructions

### Option 1: Manual Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Deploy the SageMaker endpoint:
   ```
   python deploy_endpoint.py
   ```

3. Build and run the Docker container:
   ```
   docker build -t myimage .
   docker run -d -e ENDPOINT_NAME="your-endpoint-name" -e REGION_NAME="your-region" --name mycontainer -p 80:80 myimage
   ```

4. The API is now accessible at `http://localhost:80`

5. You can invoke the endpoint with the `invoke_endpoint.py` script.

### Option 2: CloudFormation Deployment

1. Ensure you have the AWS CLI installed and configured with appropriate credentials.

2. Deploy the CloudFormation stack:
   ```
   aws cloudformation create-stack --stack-name my-sagemaker-api --template-body file://cloudformation/template.yaml --parameters ParameterKey=KeyName,ParameterValue=your-key-pair ParameterKey=VpcId,ParameterValue=vpc-xxxxxxxx ParameterKey=SubnetId,ParameterValue=subnet-xxxxxxxx
   ```

   Replace `your-key-pair` with the name of your EC2 key pair, `vpc-xxxxxxxx` with your VPC ID, and `subnet-xxxxxxxx` with your Subnet ID.

3. Wait for the stack creation to complete:
   ```
   aws cloudformation wait stack-create-complete --stack-name my-sagemaker-api
   ```

4. Retrieve the EC2 instance's public IP address:
   ```
   aws cloudformation describe-stacks --stack-name my-sagemaker-api --query "Stacks[0].Outputs[?OutputKey=='PublicIP'].OutputValue" --output text
   ```

5. You can now access the API at `http://<public-ip>:8000` (WIP: for now, you need to SSH to the instance and run the container manually)

## Usage

Once deployed, you can send POST requests to the API endpoint with your input data to receive predictions from the SageMaker model.

