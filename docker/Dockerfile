FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    htop \
    mc \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "jaxtyping>=0.2"        \
    "torchinfo>=1.8"        \
    "webdataset>=0.2"       \
    pre-commit              \
    black                   \
    isort                   \
    flake8                  \
    build                   \
    pybind11                \
    ipdb                    \
    pytest                  \
    rich                    \
    h5py                    \
    wandb                   \
    hydra-core              \
    omegaconf               \
    lightning               \
    torchmetrics            \
    cupy-cuda12x

# Install torch_scatter from github source
# ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 9.1 9.2"
# RUN pip install --no-cache-dir git+https://github.com/rusty1s/pytorch_scatter.git
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Install warpconvnet and install only the 3rdparty/cutlass submodule
RUN mkdir -p /opt/warpconvnet && cd /opt/warpconvnet && \
    git clone https://github.com/NVlabs/WarpConvNet.git && \
    cd WarpConvNet && \
    git submodule update --init 3rdparty/cutlass && \
    python -m build --wheel --no-isolation && \
    pip install dist/*.whl

# Add a non-root user with a fixed UID and GID
ARG USERNAME=wcnuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN set -eux; \
    groupadd --gid $USER_GID $USERNAME; \
    useradd --uid $USER_UID --gid $USER_GID --no-log-init -m -G video $USERNAME

# Add sudo and allow the non-root user to execute commands as root
# without a password.
RUN apt-get update && apt-get install -y \
    sudo;
RUN echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME;

WORKDIR /workspace
