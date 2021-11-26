from PIL import Image
import numpy as np
import streamlit as st
import requests
import json

# Function to Read and Convert Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image


# Uploading the Image to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])
# Checking the Format of the page
if uploadFile is not None:
    # Perform  Manipulations
    img = load_image(uploadFile)
    st.image(img)
    # st.write(img)
    st.write(":camera_with_flash: Image Uploaded Successfully !")
    # Reshape the image
    X = img.reshape(img.shape[0] * img.shape[1] * img.shape[2])
    X = X.tolist()
    X_json = json.dumps(X)
    # Call the POST
    url = "http://127.0.0.1:8000/predict/"
    data = json.dumps({
        "image_reshape": X_json,
        "height": img.shape[0],
        "width": img.shape[1],
        "color": img.shape[2]
    })
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, data, headers=headers).json()
    # e.g bruschetta
    st.write(f"I think the this sign is: {response['response']}")


else:
    st.write("Make sure you image is in JPG/PNG Format.")
