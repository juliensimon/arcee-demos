# SageMaker Proxy

## 1. Introduction

This project provides a FastAPI wrapper for Amazon SageMaker endpoints. It allows you to interact with SageMaker endpoints through a simple API interface, supporting both local and containerized deployments. The project also includes an AWS CloudFormation template for easy setup.

## General Prerequisites
* An AWS account with appropriate permissions for IAM, SageMaker, EC2, and CloudFormation.
* At least one SageMaker endpoint deployed.

## 2. Manual setup
This involves setting up and running the proxy manually on an Amazon EC2 instance.

### 2.1 Prerequisites

* An EC2 instance running in the same region as your SageMaker endpoint(s),with Amazon Linux 2023, Python 3.8+, and Docker installed.
* An IAM role attached to your EC2 instance with the `SageMakerFullAccess` policy, and a trust relationship allowing both EC2 and SageMaker to assume the role.

```1:15:trust-policy.json
{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": [
            "ec2.amazonaws.com",
            "sagemaker.amazonaws.com"
          ]
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }
```

### 2.2 Running the proxy

You can run the proxy manually or using Docker.

#### 2.2.1 Manual setup

a. Clone the repository and navigate to the project directory
   ```
   git clone https://github.com/juliensimon/sagemaker-proxy.git
   cd sagemaker-proxy
   ```
b. Install dependencies
   ```
   pip install -r requirements.txt
   ```
c. Set environment variables.

   The API_KEY must be an UUID4 string, e.g. `uuidgen`.

   The REGION_NAME must be the region of your EC2 instance and your SageMaker endpoint(s), e.g. `us-east-1`.

   ```
   export API_KEY=your_api_key
   export REGION_NAME=your_aws_region
   ```
4. Run the FastAPI app
   ```
   cd container
   python app/main.py
   ```

#### 2.2.2 Docker setup

a. Pull the Docker image
   ```
   docker pull juliensimon/sagemaker-proxy:latest
   ```
b. Run the Docker container.

   The API_KEY must be an UUID4 string, e.g. `uuidgen`.

   The REGION_NAME must be the region of your EC2 instance and your SageMaker endpoint(s), e.g. `us-east-1`.

   The container exposes port 8000, feel free to change the port mapping to your convenience.

   ```
   docker run -d -e API_KEY=your_api_key -e REGION_NAME=your_aws_region -p 8000:8000 juliensimon/sagemaker-proxy:latest
   ```

### 2.3 Running the Invoke Scripts

You can run the test scripts on your EC2 instance to test the proxy at `https://localhost:8000`. Of course, you can run them on any other machine as long as it can reach the EC2 instance: simply replace `localhost` with the public IP address or the fully qualified domain name of the EC2 instance, e.g. `ec2-123-456-789-012.us-east-1.compute.amazonaws.com`.

a. Open a new terminal on the test machine.
b. Set the API_KEY environment variable.
   ```
   export API_KEY=your_api_key
   ```
c. Run the invoke scripts:

   The scripts will list existing SageMaker endpoints and select the first one returned by the SageMaker `list_endpoints` API call.

   You can run invoke this endpoint with the `requests` library:
   ```
   python invoke_endpoint.py localhost 8000
   ```

   If the endpoint was configured to accept the OpenAI Messages API, you can also invoke it with the `openai` client:
   ```
   python invoke_endpoint_openai.py localhost 8000
   ```


## 3. CloudFormation setup

This setup involves deploying a CloudFormation stack that creates an EC2 instance running the Docker container. We use a `t3.micro`instance by default.

The EC2 instance exposes port 22 (SSH) and port 8000 (FastAPI) to the internet, allowing you to connect to the instance and to invoke the proxy from outside AWS, e.g. from your local machine as shown below.

### 3.1 Prerequisites

* An EC2 key pair.
* An Amazon VPC with a public subnet.

### 3.2 Deploy the CloudFormation stack

1. Deploy the CloudFormation stack

   You can create the stack in the AWS console or using the AWS CLI, assuming that you have set up appropriate AWS credentials on your local machine.

   The `ApiKey` parameter must be an UUID4 string, e.g. `uuidgen`.

   The `VpcId` parameter must be the ID of your VPC, e.g. `vpc-0123456789abcdefg`.

   The `SubnetId` parameter must be the ID of your public subnet, e.g. `subnet-0123456789abcdefg`.

   The `KeyName` parameter must be the name of an EC2 key pair.

   ```
   aws cloudformation create-stack --stack-name sagemaker-proxy --template-body file://cloudformation/instance.yaml --parameters ParameterKey=KeyName,ParameterValue=your_key_pair ParameterKey=VpcId,ParameterValue=vpc-xxxxxxxx ParameterKey=SubnetId,ParameterValue=subnet-xxxxxxxx ParameterKey=ApiKey,ParameterValue=your_api_key
   ```
2. Wait for the stack creation to complete:
   ```
   aws cloudformation wait stack-create-complete --stack-name sagemaker-proxy
   ```
3. Get the EC2 instance's fully qualified domain name, e.g. `ec2-123-456-789-012.us-east-1.compute.amazonaws.com`.
   ```
   aws cloudformation describe-stacks --stack-name sagemaker-proxy --query "Stacks[0].Outputs[?OutputKey=='PublicDNS'].OutputValue" --output text
   ```
4. Run the invoke script

   ```
   python invoke_endpoint.py instance_fqdn 8000
   ```

   If the endpoint was configured to accept the OpenAI Messages API, you can also invoke it with the `openai` client:
   ```
   python invoke_endpoint.py instance_fqdn 8000
   ```

## 4. Security

1. API Key Authentication: The FastAPI app uses API key authentication. Ensure the API_KEY environment variable is set securely and not exposed.

2. HTTPS: The FastAPI app uses HTTPS with a self-signed certificate. In production, replace it with a properly signed certificate.

3. IAM Roles: Use the principle of least privilege when setting up IAM roles. Only grant necessary permissions to the EC2 instance and SageMaker.

4. Network Security: Use security groups and network ACLs to restrict access to the EC2 instance and the FastAPI app.

## 5. Running Unit Tests

To run the unit tests:

1. Ensure you have pytest installed:
   ```
   pip install pytest
   ```
2. Set the API_KEY environment variable:
   ```
   export API_KEY=your_api_key
   ```
3. Navigate to the tests directory:
   ```
   cd tests
   ```
4. Run the tests:
   ```
   pytest
   ```
