from fastapi import FastAPI
import tensorflow_hub as hub
import tensorflow_text
import numpy


# for avoiding error
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = FastAPI()
# embed = hub.load("/var/folders/25/z38s61zn0dnbmjcs68r62ywc0000gn/T/tfhub_modules")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

@app.get("/ping")
def pong():
    texts = ["It is cool", "Is it cool", "It is cool", "I saw a comedy show last night.", "Yesterday, I went to the park."]
    vectors = embed(texts)
    result = vectors.numpy().tolist()
    print(type(vectors), result)
    return {"res": result}