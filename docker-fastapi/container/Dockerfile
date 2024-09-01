# Instructions:
# docker build -t myimage .
# docker run -d  -e ENDPOINT_NAME="djl-inference-2024-08-24-11-57-00-390" -e REGION_NAME="us-west-2" --name mycontainer -p 80:80 myimage

# Use a more specific and lighter base image
FROM python:alpine

# Set working directory
WORKDIR /code

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on

COPY ./requirements.txt /code/requirements.txt

COPY ./ssl/key.pem ssl/key.pem

COPY ./ssl/cert.pem ssl/cert.pem

COPY ./app app

# Install system dependencies
RUN pip install --upgrade pip
RUN pip install --upgrade -r /code/requirements.txt

CMD ["python3", "app/main.py"]

