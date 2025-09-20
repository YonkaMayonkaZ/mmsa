FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY docker/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Install CMU Multimodal SDK
RUN git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git /tmp/mmsdk && \
    cd /tmp/mmsdk && \
    pip install . && \
    rm -rf /tmp/mmsdk

# Create necessary directories
RUN mkdir -p /workspace/data/raw /workspace/data/processed /workspace/data/cache \
    /workspace/outputs/models /workspace/outputs/logs /workspace/outputs/metrics \
    /workspace/outputs/visualizations

# Copy application code
COPY . /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0
ENV WANDB_MODE=offline

# Set entrypoint
ENTRYPOINT ["/bin/bash"]