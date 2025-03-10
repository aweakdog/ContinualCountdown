FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
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
    conda run -n zero pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n zero pip install vllm==0.6.3 && \
    conda run -n zero pip install ray && \
    conda run -n zero pip install -e . && \
    conda run -n zero pip install flash-attn --no-build-isolation && \
    conda run -n zero pip install wandb IPython matplotlib && \
    conda run -n zero pip install datasets

# Create data directory
RUN mkdir -p /data/countdown

# Set default command to activate conda environment
SHELL ["conda", "run", "-n", "zero", "/bin/bash", "-c"]

# Set working directory for data
WORKDIR /data/countdown

# Download and prepare the dataset when container starts
CMD ["conda", "run", "-n", "zero", "python", "/app/examples/data_preprocess/countdown.py", "--local_dir", "/data/countdown"]
