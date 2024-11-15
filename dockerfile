FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install fastapi uvicorn

RUN pip3 install \
    Pillow \
    torch \
    torchvision \
    transformers \
    sentencepiece \
    accelerate \
    bitsandbytes \
    python-multipart \
    decord

COPY . /app

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8080 && sleep infinity"]