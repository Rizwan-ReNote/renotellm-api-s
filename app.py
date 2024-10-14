import logging
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from PIL import Image
from typing import Optional, Union
import torch
from transformers import AutoModel, AutoTokenizer
import ast
 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 
 
# Function to clear CUDA cache
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA cache cleared")
app = FastAPI()
clear_cuda_cache()
access_token = "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"
 
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
model.eval()
model.config.use_cache = False
 
@app.post("/vllm")
async def vllm(
    image: UploadFile = File(None),  # Make the file upload optional
    question: Union[str, list] = Form(...),
    sampling: Optional[bool] = Form(True),
    temperature: Optional[float] = Form(0.7),
    stream: Optional[bool] = Form(False),
    system_prompt: Optional[str] = Form(''),
):
    clear_cuda_cache()
    try:
        if image:
            image = Image.open(image.file).convert('RGB')
        clear_cuda_cache()
       
        # Check if question is a string and try to parse it as a list
        if isinstance(question, str):
            try:
                # Attempt to convert the string to a Python list if it's in the correct format
                parsed_question = ast.literal_eval(question)
                if isinstance(parsed_question, list):
                    msgs = parsed_question
                else:
                    msgs = [{'role': 'user', 'content': question}]
            except (ValueError, SyntaxError):
                # If it can't be parsed, treat it as a regular string
                msgs = [{'role': 'user', 'content': question}]
        elif isinstance(question, list):
            msgs = question
        else:
            return {"error": "Invalid input for question. Expected string or list."}
       
        logging.info(f"Input: {msgs}")
        clear_cuda_cache()
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=sampling,
            temperature=temperature,
            stream=stream,
            system_prompt=system_prompt
        )
       
        logging.info(f"Model Response: {res}")
        clear_cuda_cache()
        return res
    except Exception as e:
        logging.error(str(e))
        return {"error": str(e)}