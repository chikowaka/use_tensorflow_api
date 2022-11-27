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
    tweet: List[str]
    idea: List[str]

@app.post("/embed")
def embed(req: ReqModel):
    print('req',req)
    tweet_texts = req.tweet
    tweet_vectors = use_embed(tweet_texts)
    tweet_result = tweet_vectors.numpy().tolist()
    
    idea_texts = req.idea
    idea_vectors = use_embed(idea_texts)
    idea_result = idea_vectors.numpy().tolist()
    return {"tweet_vectors": tweet_result, "idea_vectors":idea_result}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
