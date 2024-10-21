# # Use an official NVIDIA image with CUDA and Python pre-installed
# FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# # Set the working directory in the container
# WORKDIR /app

# # Install sudo and other necessary packages
# RUN apt-get update && apt-get install -y sudo

# # Install Python and pip (if not already included in the base image)
# RUN apt-get install -y python3 python3-pip

# RUN pip3 install --upgrade pip

# # Install FastAPI and Uvicorn
# RUN pip3 install fastapi uvicorn

# # Install additional dependencies
# RUN pip3 install \
#     Pillow \
#     uvicorn \
#     torch \
#     torchvision \
#     transformers \
#     sentencepiece \
#     accelerate \
#     bitsandbytes \
#     python-multipart \
#     decord \
#     packaging \
#     flash-attn  
# # Copy the current directory contents into the container at /app
# COPY . .

# # Make port 8000 available to the world outside this container
# EXPOSE 8080

# # Run the FastAPI application using Uvicorn server
# # CMD ["uvicorn", "mainv2:app", "--host", "0.0.0.0", "--port", "8080"]
# CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 && sleep infinity"]

#############################################################################################################

# # Use an official NVIDIA image with CUDA and Python pre-installed
# FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# # Set the working directory in the container
# WORKDIR /app

# # Install sudo and other necessary packages
# RUN apt-get update && apt-get install -y sudo

# # Install Python and pip (if not already included in the base image)
# RUN apt-get install -y python3 python3-pip

# # Install FastAPI and Uvicorn
# RUN pip3 install fastapi uvicorn

# # Install additional dependencies
# RUN pip3 install \
#     Pillow \
#     torch \
#     torchvision \
#     transformers \
#     sentencepiece \
#     accelerate \
#     bitsandbytes \
#     python-multipart

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Make port 8000 available to the world outside this container
# EXPOSE 8080

# # Run the FastAPI application using Uvicorn server
# CMD ["sh", "-c", "uvicorn mainv2:app --host 0.0.0.0 --port 8080 && sleep infinity"]





###################################################################################################
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
    python-multipart

# Pre-download the model and tokenizer
RUN python3 -c "\
    from transformers import AutoModel, AutoTokenizer; \
    access_token = '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'; \
    model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True, use_auth_token=access_token); \
    tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True, use_auth_token=access_token)"

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the FastAPI application using Uvicorn server
CMD ["sh", "-c", "uvicorn mainv2:app --host 0.0.0.0 --port 8080 && sleep infinity"]

