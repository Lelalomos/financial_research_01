# Use slim Python base image (much smaller than nvidia/cuda)
# CUDA support comes from PyTorch wheels in requirements.txt
FROM python:3.11-slim

# Install minimal system dependencies
# - build tools for TA-Lib compilation
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libgomp1 \
    tmux \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Create non-root user with same UID as host user (commonly 1000)
# This prevents permission issues with mounted volumes
RUN useradd -m -u 1000 appuser

# Set working directory (creates /app as root)
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code and change ownership
COPY --chown=appuser:appuser . .

# Create directories with correct ownership
RUN mkdir -p data/raw data/processed data/splits data/external logs models/checkpoints outputs reports && \
    chown -R appuser:appuser /app

# Make scripts executable
RUN chmod +x scripts/*.sh 2>/dev/null || true

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0+PTX"

# Keep container running for interactive use
CMD ["tail", "-f", "/dev/null"]
