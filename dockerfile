# Use an official NVIDIA image with CUDA and Python pre-installed
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
 
# Set the working directory in the container
WORKDIR /app
 
# Install sudo and other necessary packages
RUN apt-get update && apt-get install -y sudo
 
# Install Python and pip (if not already included in the base image)
RUN apt-get install -y python3 python3-pip
 
# Install FastAPI and Uvicorn
RUN pip3 install fastapi uvicorn
 
# Install additional dependencies
RUN pip3 install \
    Pillow \
    torch \
    torchvision \
    transformers \
    sentencepiece \
    accelerate \
    bitsandbytes \
    python-multipart \
    flash-attn
 
# Copy the current directory contents into the container at /app
COPY . /app
 
# Make port 8000 available to the world outside this container
EXPOSE 8000
 
# Run the FastAPI application using Uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]