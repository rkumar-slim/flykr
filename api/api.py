from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import  tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from flykr.utils import model_label_prediction
from skimage import color

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = load_model('asl_model.h5')
labels = np.load('asl_class_names.npy')


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
    response_reshape_1 = response.reshape(
        (image.height, image.width, image.color))
    response_reshape_2 = response_reshape_1[:, :, :3]
    # Resize the image :warning: WITHOUT PAD
    response_reshape = tf.image.resize(response_reshape_2, (128, 128))
    # response_reshape = np.array(response_reshape).reshape(
    #     -1, 128, 128, 3)
    print(response_reshape.shape)
    # Load the model
    letter = model_label_prediction(model, labels, response_reshape)
    #letter = response_reshape_1[1, :]

    return {"response": str(letter)}
