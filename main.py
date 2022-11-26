from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow_hub as hub
import tensorflow_text
import numpy
from typing import List

# for avoiding error
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()
# embed = hub.load("/var/folders/25/z38s61zn0dnbmjcs68r62ywc0000gn/T/tfhub_modules")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

class PongModel(BaseModel):
    text: List[str]

@app.post("/ping")
def pong(req: PongModel):
    print(req.text)
    texts = req.text
    vectors = embed(texts)
    result = vectors.numpy().tolist()
    print(type(vectors), result)
    return {"res": result}