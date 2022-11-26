from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow_hub as hub
import tensorflow_text
import numpy
from typing import List
import uvicorn

# for avoiding error
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()
# embed = hub.load("/var/folders/25/z38s61zn0dnbmjcs68r62ywc0000gn/T/tfhub_modules")
use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

class ReqModel(BaseModel):
    text: List[str]

@app.post("/embed")
def embed(req: ReqModel):
    print('req',req.text)
    texts = req.text
    vectors = use_embed(texts)
    result = vectors.numpy().tolist()
    print(type(vectors), result)
    return {"res": result}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
