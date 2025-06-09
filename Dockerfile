# Use an official Python runtime as a parent image
# Use a base image with Python and CUDA/cuDNN pre-installed from NVIDIA
# This ensures GPU support is ready. Choose a CUDA version compatible with your YOLO/PyTorch needs.
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=interactive

# Update system and install necessary packages (ffmpeg for video processing, git for potential future use)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default (ensure it's installed and symlinked correctly)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements.txt and install Python dependencies first
# This improves Docker cache efficiency
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application files and folders into the container
# This includes main.py, best.pt, and the entire utils directory
COPY main.py ./
COPY best.pt ./
COPY utils/ ./utils/

# Expose the port that FastAPI will run on
EXPOSE 8080

# Command to run your FastAPI application with Uvicorn
# 0.0.0.0 makes the app accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]