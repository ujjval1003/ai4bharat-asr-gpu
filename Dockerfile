# Base Image (CUDA 12.6 for RTX 3050)
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libasound2-dev \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Workdir
WORKDIR /workspace

# Copy project files
COPY . .

# Run environment setup
RUN chmod +x setup.sh && bash setup.sh

# Expose Gradio UI
EXPOSE 7860

# Default command
CMD ["/bin/bash"]