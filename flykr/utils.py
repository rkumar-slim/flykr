import numpy as np

def model_label_prediction(model, labels, X_test):
    X_test = np.reshape(X_test,
                        (-1, model.input_shape[1], model.input_shape[2], 3))
    # need to rescale if numbers are not between 0 and 1
    if X_test.max() > 1:
        X_test = X_test / 255.

    prediction = model.predict(X_test)
    return labels[np.argmax(prediction)]


def scale(image, label):
    return image / 255., label
