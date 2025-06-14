# Use NVIDIA's Ubuntu 22.04-based CUDA 11.8 image as the base
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies and Python 3.9
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/python3.9-pip 1

# Copy requirements.txt to install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/data /app/weights /app/result /app/modules

# Download SAM model weights
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P /app/weights

# Copy project files
COPY main.py .
COPY modules/ /app/modules/

# Set environment variables for CUDA and PYTHONPATH
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/app:${PYTHONPATH}

# Command to run the main script
CMD ["python3", "main.py"]