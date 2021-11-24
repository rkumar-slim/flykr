from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world! Welcome to our Sign Language Demo :)"}


@app.get("/predict")
def predict(picture):

    picture = picture[1:-1]
    picture = np.array(list((map(lambda x: float(x),
                       picture.split(','))))).reshape(1, 100, 100, 1)

    model = load_model('model.h5')

    prediction = model.predict(picture).flatten()
    prediction = (prediction > 0.5)

    labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd',
        'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
        's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    ]

    letter = ""

    for i in range(len(prediction)):
        if prediction[i]:
            letter = labels[i]

    return {"prediction": letter}
