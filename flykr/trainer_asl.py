import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import image_dataset_from_directory
from flykr.utils import scale


class Trainer(object):

    def __init__(self, image_size_x, image_size_y, data_dir):
        # Path of training and validation data
        self.data_dir = data_dir

        # Image size [image_size_x, image_size_y] required by neural network
        self.image_size_x = image_size_x
        self.image_size_y = image_size_y

        # Load in train dataset. Labels are automatically created as images are organized into subfolders
        # with folder name being the label.
        self.train_dataset = image_dataset_from_directory(
            self.data_dir,
            shuffle=True,
            color_mode="rgb",
            label_mode='categorical',
            validation_split=0.3,
            batch_size=64,
            labels='inferred',
            image_size=(self.image_size_x, self.image_size_y),
            subset="training",
            seed=123)

        # Load in validation dataset
        self.val_dataset = image_dataset_from_directory(
            self.data_dir,
            shuffle=True,
            color_mode="rgb",
            label_mode='categorical',
            validation_split=0.3,
            batch_size=16,
            labels='inferred',
            image_size=(self.image_size_x, self.image_size_y),
            subset="validation",
            seed=123)

        # Label names created during the loading of train ds. This is saved down along with the final model.
        self.class_names = self.train_dataset.class_names

        self.model = None
        self.transfer_model = VGG16(weights="imagenet",
                                    include_top=False,
                                    input_shape=(self.image_size_x,
                                                 self.image_size_y, 3))

    def set_structure(self, trainable = False):

        model = models.Sequential([
            self.transfer_model,
            layers.Flatten(),
            layers.Dense(500, activation='relu'),
            layers.Dense(250, activation='relu'),
            layers.Dense(125, activation='relu'),
            layers.Dense(len(self.class_names), activation='softmax')
        ])

        return model

    def set_transfer(self, trainable = False):
        self.transfer_model.trainable = trainable


    def compile_model(self, learning_rate=0.001):

        opt = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])


    def run(self, epochs = 50):
        self.train_dataset = self.train_dataset.map(scale)
        self.val_dataset = self.val_dataset.map(scale)

        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights=True)

        self.model.fit(x=self.train_dataset,
                       epochs=epochs,
                       validation_data=self.val_dataset,
                       shuffle = True,
                       initial_epoch= 0,
                       callbacks=[es])

    def evaluate(self, X_test, y_test):
        pass

    def save_down_model(self):
        self.model.save('asl_model.h5')
        np.save('asl_class_names.npy', self.class_names)

        print("Saved down asl model locally.")



if __name__ == "__main__":

    data_filepath = './raw_data/asl_data/'
    image_size_x = 128
    image_size_y = 128

    # Train and save model, locally and
    trainer = Trainer(image_size_x, image_size_y, data_filepath)
    trainer.model = trainer.set_structure()
    trainer.set_transfer()
    trainer.compile_model()
    trainer.run(epochs=1)
    trainer.set_transfer(trainable=True)
    trainer.compile_model(learning_rate=1e-5)
    trainer.run(epochs=1)

    trainer.save_down_model()
