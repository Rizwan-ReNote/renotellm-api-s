# import gc
# import logging
# import os
# import tempfile
# from typing import List, Optional

# import torch
# import uvicorn
# from decord import VideoReader, cpu
# from fastapi import FastAPI, File, Form, UploadFile
# from PIL import Image
# from transformers import AutoModel, AutoTokenizer

# app = FastAPI()

# torch.set_grad_enabled(False)

# # Directory to store model locally
# MODEL_DIR = "./local_model"

# # Function to clear CUDA cache


# def clear_cuda_cache():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         gc.collect()

# # Function to load or download the model


# def load_model_and_tokenizer():
#     if not os.path.exists(MODEL_DIR):
#         os.makedirs(MODEL_DIR)

#     model_path = os.path.join(MODEL_DIR, 'MiniCPM-V-2_6-int4')

#     # Check if model weights exist locally
#     if not os.path.exists(model_path):
#         logging.info("Downloading model and tokenizer...")
#         model = AutoModel.from_pretrained(
#             'openbmb/MiniCPM-V-2_6-int4',
#             trust_remote_code=True,
#             use_auth_token="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"
#         )
#         tokenizer = AutoTokenizer.from_pretrained(
#             'openbmb/MiniCPM-V-2_6-int4',
#             trust_remote_code=True,
#             use_auth_token="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"
#         )

#         # Save the model and tokenizer locally
#         model.save_pretrained(model_path)
#         tokenizer.save_pretrained(model_path)
#     else:
#         logging.info("Loading model and tokenizer from local directory...")
#         model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_path, trust_remote_code=True)

#     return model, tokenizer


# # Load model and tokenizer (either from local or downloading if not available)
# model, tokenizer = load_model_and_tokenizer()

# model.config.use_cache = False
# model.eval()


# @app.post("/vllm")
# async def vllm(
#     images: Optional[List[UploadFile]] = File(None),  # Allow 0 to n images
#     question: str = Form(...),
#     sampling: Optional[bool] = Form(True),
#     temperature: Optional[float] = Form(0.7),
#     max_tokens: Optional[int] = Form(250),  # Default max token limit
#     min_tokens: Optional[int] = Form(50),   # Default min token limit
#     stream: Optional[bool] = Form(False),
#     system_prompt: Optional[str] = Form(''),
# ):
#     try:
#         # Process images if provided, otherwise set to an empty list
#         img_list = []
#         if images:
#             for image in images:
#                 img = Image.open(image.file).convert('RGB')
#                 img_list.append(img)

#         # Prepare messages, including images if any
#         content = img_list + [question] if img_list else [question]
#         msgs = [{'role': 'user', 'content': content}]

#         # Model response with dynamic token limits
#         response = model.chat(
#             image=None,  # Model accepts images in 'msgs' content, not directly
#             msgs=msgs,
#             tokenizer=tokenizer,
#             sampling=sampling,
#             temperature=temperature,
#             max_tokens=max_tokens,  # Apply max token limit
#             min_tokens=min_tokens,  # Apply min token limit
#             stream=stream,
#             system_prompt=system_prompt
#         )

#         return {"answer": response}

#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return {"error": str(e)}

#     finally:
#         clear_cuda_cache()


# # Constants
# MAX_NUM_FRAMES = 64  # Adjust for CUDA memory limits


# def encode_video(video_path: str):
#     """Encodes video into frames suitable for model input."""
#     def uniform_sample(l, n):
#         gap = len(l) / n
#         idxs = [int(i * gap + gap / 2) for i in range(n)]
#         return [l[i] for i in idxs]

#     vr = VideoReader(video_path, ctx=cpu(0))
#     sample_fps = round(vr.get_avg_fps() / 1)  # FPS
#     frame_idx = [i for i in range(0, len(vr), sample_fps)]
#     if len(frame_idx) > MAX_NUM_FRAMES:
#         frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
#     frames = vr.get_batch(frame_idx).asnumpy()
#     frames = [Image.fromarray(v.astype('uint8')) for v in frames]
#     logging.info(f'Number of frames extracted: {len(frames)}')
#     return frames


# @app.post("/video_description")
# async def video_description(
#     video: UploadFile = File(...),
#     question: str = Form("Describe the video"),
#     max_slice_nums: Optional[int] = Form(2),
#     use_image_id: Optional[bool] = Form(False)
# ):
#     try:
#         # Save the uploaded video to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
#             tmp_video.write(video.file.read())
#             tmp_video_path = tmp_video.name

#         # Encode video into frames
#         frames = encode_video(tmp_video_path)

#         # Prepare the message content for the model
#         msgs = [{'role': 'user', 'content': frames + [question]}]

#         # Set parameters for model decoding
#         params = {
#             "use_image_id": use_image_id,
#             "max_slice_nums": max_slice_nums
#         }

#         # Get the model's answer
#         answer = model.chat(
#             image=None,
#             msgs=msgs,
#             tokenizer=tokenizer,
#             **params
#         )

#         return {"answer": answer}

#     except Exception as e:
#         logging.error(f"Error processing video: {e}")
#         return {"error": str(e)}

#     finally:
#         if 'tmp_video_path' in locals():
#             os.remove(tmp_video_path)  # Ensure temporary file is deleted

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)

import asyncio
import gc
import logging
import os
import tempfile
from typing import List, Optional

import torch
from decord import VideoReader, cpu
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

torch.set_grad_enabled(False)

# Load environment variables from .env file
load_dotenv()

# Get the Hugging Face token from environment variables
HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')

# Directory to store model locally
MODEL_DIR = "./local_model"

def clear_cuda_cache():
    """Clears the CUDA cache to free up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        logging.debug("CUDA cache cleared.")

def load_model_and_tokenizer():
    """Loads or downloads the model and tokenizer."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logging.debug(f"Created model directory at {MODEL_DIR}.")

    model_path = os.path.join(MODEL_DIR, 'MiniCPM-V-2_6-int4')

    # Check if model weights exist locally
    if not os.path.exists(model_path):
        logging.info("Downloading model and tokenizer...")
        model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-V-2_6-int4',
            trust_remote_code=True,
            use_auth_token=HUGGINGFACE_TOKEN
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'openbmb/MiniCPM-V-2_6-int4',
            trust_remote_code=True,
            use_auth_token=HUGGINGFACE_TOKEN
        )

        # Save the model and tokenizer locally
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logging.info("Model and tokenizer downloaded and saved locally.")
    else:
        logging.info("Loading model and tokenizer from local directory...")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        logging.info("Model and tokenizer loaded from local directory.")

    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()
model.config.use_cache = False
model.eval()

MAX_NUM_FRAMES = 64  # Adjust for CUDA memory limits

def encode_video(video_path: str):
    """Encodes video into frames suitable for model input."""
    logging.debug(f"Encoding video at path: {video_path}")
    
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = max(1, round(vr.get_avg_fps()))
    frame_idx = list(range(0, len(vr), sample_fps))
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(frame.astype('uint8')) for frame in frames]
    logging.info(f"Number of frames extracted: {len(frames)}")
    return frames

@app.post("/generate")
async def generate(
    images: Optional[List[UploadFile]] = File(None),
    question: str = Form(...),
    sampling: bool = Form(True),
    temperature: float = Form(0.7),
    max_tokens: int = Form(250),
    min_tokens: int = Form(50),
    stream: bool = Form(False),
    system_prompt: str = Form(''),
):
    """Generates a response based on provided images and question."""
    try:
        logging.info("Received request at /generate")
        img_list = []
        if images:
            for image in images:
                img = await asyncio.to_thread(Image.open, image.file)
                img = img.convert('RGB')
                img_list.append(img)
                logging.debug(f"Processed image: {image.filename}")

        content = img_list + [question] if img_list else [question]
        msgs = [{'role': 'user', 'content': content}]

        response = await asyncio.to_thread(
            model.chat,
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=sampling,
            temperature=temperature,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            stream=stream,
            system_prompt=system_prompt
        )

        logging.info("Generated response successfully.")
        return {"answer": response}

    except Exception as e:
        logging.error(f"Error in /generate: {e}")
        return {"error": str(e)}

    finally:
        clear_cuda_cache()

@app.post("/videos/description")
async def describe_video(
    video: UploadFile = File(...),
    question: str = Form("Describe the video"),
    max_slice_nums: int = Form(2),
    use_image_id: bool = Form(False)
):
    """Generates a description for the provided video."""
    try:
        logging.info("Received request at /videos/description")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            video_bytes = await video.read()
            tmp_video.write(video_bytes)
            tmp_video_path = tmp_video.name
            logging.debug(f"Video saved to temporary file: {tmp_video_path}")

        frames = await asyncio.to_thread(encode_video, tmp_video_path)

        msgs = [{'role': 'user', 'content': frames + [question]}]

        params = {
            "use_image_id": use_image_id,
            "max_slice_nums": max_slice_nums
        }

        answer = await asyncio.to_thread(
            model.chat,
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params
        )

        logging.info("Generated video description successfully.")
        return {"answer": answer}

    except Exception as e:
        logging.error(f"Error in /videos/description: {e}")
        return {"error": str(e)}

    finally:
        if 'tmp_video_path' in locals():
            os.remove(tmp_video_path)
            logging.debug(f"Deleted temporary video file: {tmp_video_path}")
        clear_cuda_cache()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
