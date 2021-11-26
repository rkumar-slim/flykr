from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import  tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = load_model('asl_model.h5')
label_binarizer = joblib.load("asl_labelbinarizer.h5")


@app.get("/")
def index():
    return {"greeting": "Hello world! Welcome to our Sign Language Demo :)"}


class Item(BaseModel):
    image_reshape: str
    height: int
    width: int
    color: int


@app.post("/predict")
async def predict(image: Item):
    # Get the image from the upload
    response = np.array(json.loads(image.image_reshape))
    response_reshape = response.reshape(
        (image.height, image.width, image.color))
    # Resize the image :warning: WITHOUT PAD
    response_reshape = tf.image.resize(response_reshape, [100, 100])
    response_reshape = np.array(response_reshape[:, :, :3]).reshape(
        -1, 100, 100, 3)
    print(response_reshape.shape)
    # Load the model
    prediction = (model.predict(response_reshape) > 0.5) * 1.0
    letter = label_binarizer.inverse_transform(prediction)[0]

    return {"response": letter}
