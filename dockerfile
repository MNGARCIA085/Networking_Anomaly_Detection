# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------
# Base: common runtime dependencies and installed package
# ---------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Select exact versions and the PyTorch wheel repository.
ARG TORCH_VERSION=2.11.0
ARG LIGHTNING_VERSION=2.6.6
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Make the selected versions constraints for subsequent
# pip installations, including installation of this package.
RUN printf 'torch==%s\nlightning==%s\n' \
    "${TORCH_VERSION}" \
    "${LIGHTNING_VERSION}" \
    > /tmp/constraints.txt

ENV PIP_CONSTRAINT=/tmp/constraints.txt

# Copy package metadata and source.
COPY pyproject.toml .
COPY src ./src

# Install the selected PyTorch build first.
RUN pip install --no-cache-dir \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}"

# Install the package and its declared runtime dependencies.
# The constraints preserve the selected Torch and Lightning versions.
RUN pip install --no-cache-dir .

# ---------------------------------------------------------
# Inference: runtime package only
# ---------------------------------------------------------
FROM base AS inference

# Artifacts are provided at runtime, not baked into the image.
# Mount model and preprocessing artifacts where your inference
# application expects to find them.

CMD ["python", "-c", "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"]


# ---------------------------------------------------------
# Train: training tools, scripts, and Hydra configuration
# ---------------------------------------------------------
FROM base AS train

COPY config ./config
COPY scripts ./scripts

# Install the optional training dependencies.
# The same constraints apply, so Torch and Lightning stay selected.
RUN pip install --no-cache-dir ".[train]"

CMD ["bash"]