# import gc
# import logging
# from typing import Optional

# import torch
# import uvicorn
# from fastapi import FastAPI, File, Form, UploadFile
# from PIL import Image
# from pydantic import BaseModel
# from transformers import AutoModel, AutoTokenizer

# app = FastAPI()

# torch.set_grad_enabled(False)

# access_token = "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"

# model = AutoModel.from_pretrained(
#     'openbmb/MiniCPM-V-2_6-int4',
#     trust_remote_code=True,
#     use_auth_token=access_token
# )
# tokenizer = AutoTokenizer.from_pretrained(
#     'openbmb/MiniCPM-V-2_6-int4',
#     trust_remote_code=True,
#     use_auth_token=access_token
# )

# model.config.use_cache = False
# model.eval()


# def clear_cuda_cache():
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
#         gc.collect()


# @app.post("/vllm")
# async def vllm(
#     image: UploadFile = File(None),
#     question: str = Form(...),
#     sampling: Optional[bool] = Form(True),
#     temperature: Optional[float] = Form(0.7),
#     stream: Optional[bool] = Form(False),
#     system_prompt: Optional[str] = Form(''),
# ):
#     try:
#         img = None
#         if image:
#             img = Image.open(image.file).convert('RGB')

#         msgs = [{'role': 'user', 'content': question}]

#         # Model response
#         response = model.chat(
#             image=img,
#             msgs=msgs,
#             tokenizer=tokenizer,
#             sampling=sampling,
#             temperature=temperature,
#             stream=stream,
#             system_prompt=system_prompt
#         )

#         return response

#     except Exception as e:
#         logging.error(f"Error: {str(e)}")
#         return {"error": str(e)}

#     finally:
#         clear_cuda_cache()


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)

#############################################################################
import gc
import logging
import os
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

torch.set_grad_enabled(False)

# Directory to store model locally
MODEL_DIR = "./model_weights"

# Function to clear CUDA cache


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

# Function to load or download the model


def load_model_and_tokenizer():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, 'MiniCPM-V-2_6-int4')

    # Check if model weights exist locally
    if not os.path.exists(model_path):
        logging.info("Downloading model and tokenizer...")
        model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-V-2_6-int4',
            trust_remote_code=True,
            use_auth_token="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"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'openbmb/MiniCPM-V-2_6-int4',
            trust_remote_code=True,
            use_auth_token="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"
        )

        # Save the model and tokenizer locally
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    else:
        logging.info("Loading model and tokenizer from local directory...")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

    return model, tokenizer


# Load model and tokenizer (either from local or downloading if not available)
model, tokenizer = load_model_and_tokenizer()

model.config.use_cache = False
model.eval()


@app.post("/vllm")
async def vllm(
    image: UploadFile = File(None),
    question: str = Form(...),
    sampling: Optional[bool] = Form(True),
    temperature: Optional[float] = Form(0.7),
    stream: Optional[bool] = Form(False),
    system_prompt: Optional[str] = Form(''),
):
    try:
        img = None
        if image:
            img = Image.open(image.file).convert('RGB')

        msgs = [{'role': 'user', 'content': question}]

        # Model response
        response = model.chat(
            image=img,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=sampling,
            temperature=temperature,
            stream=stream,
            system_prompt=system_prompt
        )

        return response

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return {"error": str(e)}

    finally:
        clear_cuda_cache()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)
