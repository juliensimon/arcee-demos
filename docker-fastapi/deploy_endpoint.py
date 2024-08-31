
"""Module for deploying a model to a SageMaker endpoint."""

import boto3
import sagemaker
from sagemaker.djl_inference.model import DJLModel

if __name__ == "__main__":

    REGION = 'us-west-2'
    MODEL_ID = "arcee-ai/Arcee-Scribe"
    INSTANCE_TYPE = "ml.g5.2xlarge"

    # Configure AWS session and SageMaker session
    boto3.setup_default_session(region_name=REGION)
    sagemaker_session = sagemaker.Session(boto3.Session(region_name=REGION))

    # Prerequisite: make sure that the IAM role for the EC2 instance has the
    # necessary permissions to create and manage SageMaker resources
    # (e.g., SageMakerFullAccess) as well as a trust policy that
    # allows SageMaker to assume the role, see trust-policy.json

    # Get the IAM execution role for SageMaker
    role = sagemaker.get_execution_role()

    # Create a DJLModel instance with specified configuration
    model = DJLModel(
        model_id=MODEL_ID,
        role=role,
        env={
            # Use bfloat16 data type for improved performance
            "OPTION_DTYPE": "bf16",
            # Set maximum sequence length
            "OPTION_MAX_MODEL_LEN": "4096",
            # Allow execution of remote code
            "OPTION_TRUST_REMOTE_CODE": "true",
            # Enable rolling batch processing using vllm
            "OPTION_ROLLING_BATCH": "vllm",
            # Use maximum tensor parallelism
            "TENSOR_PARALLEL_DEGREE": "max",
            # Set rolling batch size
            "OPTION_MAX_ROLLING_BATCH_SIZE": "16",
        },
        sagemaker_session=sagemaker_session
    )

    # Deploy the model to create a SageMaker endpoint
    predictor = model.deploy(
        # Start with one instance
        initial_instance_count=1,
        # Use the specified instance type
        instance_type=INSTANCE_TYPE,
        # Set timeout for container startup health check
        container_startup_health_check_timeout=300
    )

    # Print the name of the created endpoint
    print(f"Endpoint name: {predictor.endpoint_name}")
