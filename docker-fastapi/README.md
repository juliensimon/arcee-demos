
This project provides a Docker-based FastAPI wrapper for a SageMaker endpoint.

## Description

This application deploys an Arcee model to Amazon SageMaker and creates a FastAPI server that acts as a wrapper for the SageMaker endpoint. It enables users to make predictions using the deployed model through a simple API interface.

## Setup Instructions


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
