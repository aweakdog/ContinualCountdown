FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV VLLM_ATTENTION_BACKEND=XFORMERS
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
ENV VLLM_USE_CUDA_GRAPH=0
ENV TRANSFORMERS_OFFLINE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV VLLM_ATTENTION_BACKEND=XFORMERS
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"
ENV VLLM_USE_CUDA_GRAPH=0
ENV TRANSFORMERS_OFFLINE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    ninja-build \
    build-essential \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Create conda environment and install dependencies
WORKDIR /app
COPY . /app/

RUN conda create -n zero python=3.9 -y && \
    conda run -n zero pip install numpy && \
    #conda run -n zero pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n zero pip install vllm==0.5.4 && \
    conda run -n zero pip install nvidia-cublas-cu12==12.4.5.8 && \
    conda run -n zero pip install ray && \
    conda run -n zero pip install -e . && \
    conda run -n zero pip install flash-attn --no-build-isolation && \
    conda run -n zero pip install wandb IPython matplotlib && \
    conda run -n zero pip install datasets rich

# Create data directory
RUN mkdir -p /data/countdown

# Set default command to activate conda environment
SHELL ["conda", "run", "-n", "zero", "/bin/bash", "-c"]

# Set working directory for data
WORKDIR /data/countdown

# Set the default command to run our data viewer
CMD ["conda", "run", "-n", "zero", "python", "/app/tmp/show_data.py"]
