# Start from a specific NVIDIA CUDA image with development tools on Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set working directory
WORKDIR /workspace

# Install system dependencies, including git and software-properties-common for PPAs
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA and install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy requirements first (for better caching)
COPY ./docker/requirements.txt /tmp/requirements.txt

# Upgrade pip and install all Python dependencies from the requirements file
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements.txt


# Copy application code
COPY . /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV WANDB_MODE=offline

# Set default command to keep container alive for exec
CMD ["tail", "-f", "/dev/null"]