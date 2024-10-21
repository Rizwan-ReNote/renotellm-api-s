from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
import logging
from typing import Optional
import gc
import uvicorn
import torch
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

torch.set_grad_enabled(False)

access_token = "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"

model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-V-2_6-int4',
    trust_remote_code=True,
    use_auth_token=access_token
)
tokenizer = AutoTokenizer.from_pretrained(
    'openbmb/MiniCPM-V-2_6-int4',
    trust_remote_code=True,
    use_auth_token=access_token
)

model.config.use_cache = False
model.eval()


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


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

