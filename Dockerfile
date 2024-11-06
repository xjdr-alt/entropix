FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .

# Install rust for tiktoken
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="/root/.local/bin:/root/.cargo/bin:$PATH" && \
    poetry install --with server

# Copy only the entropix package code
COPY entropix/ entropix/

# Set default environment variables that can be overridden
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_FLAGS=--xla_gpu_enable_command_buffer= \
    API_KEYS='sk-test-key' \
    ALLOWED_ORIGINS='*' \
    PYTHONPATH=. \
    PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with proper concurrency settings for JAX
CMD ["/root/.local/bin/poetry", "run", "uvicorn", "entropix.server_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
