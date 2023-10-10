# Use the specified base image
FROM python:3.7.5
# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED 1
# Create and set the working directory
# RUN mkdir /code
# WORKDIR /code
# Upgrade pip and install git
RUN apt-get update && \
    apt-get install -y git && \
    pip install --upgrade pip \
    pip install -r requirements.txt \
    pip install dgl-cu100==0.4.3 \
    pip install pandas \
    pip install scikit-learn \
    pip install argparse
# Copy current directory contents to the container
# COPY . /code/