import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import joblib

class Trainer(object):

    def __init__(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=0)
        self.model = self.set_structure()

    def set_structure(self):
        self.label_binarizer=LabelBinarizer()
        self.y_train = self.label_binarizer.fit_transform(self.y_train)
        self.y_val = self.label_binarizer.fit_transform(self.y_val)

        transfer_model = VGG16(weights="imagenet",
            include_top=False,
            input_shape=self.X_train[0].shape)

        transfer_model.trainable = False

        model = models.Sequential([
            transfer_model,
            layers.Flatten(),
            layers.Dense(500, activation='relu'),
            layers.Dense(36, activation='softmax')
        ])


        opt = optimizers.Adam(learning_rate=1e-4)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])


        return model

    def run(self, epochs = 50):

        es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, verbose=1, restore_best_weights=True)

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0.1,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0.1,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest',
                                  validation_split=0.3)


        self.model.fit(train_datagen.flow(self.X_train,self.y_train,batch_size=64,subset='training'),
                    epochs=epochs,
                    validation_data = train_datagen.flow(self.X_val,self.y_val,batch_size=8,subset='validation'),
                    callbacks=[es])

    def evaluate(self, X_test, y_test):
        pass

    def save_down_model(self):
        self.model.save('asl_model.h5')
        joblib.dump(self.label_binarizer, 'asl_labelbinarizer.h5')
        print("Saved down asl model locally.")



if __name__ == "__main__":
    # Get and clean data
    data = pd.read_csv('./raw_data/asl_data_compressed_color.csv')
    label = pd.read_csv('./raw_data/asl_data_label.csv')

    data = np.array(data).reshape(-1, 100, 100, 3)

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        label,
                                                        test_size=0.1)
    # Train and save model, locally and
    trainer = Trainer(X_train, y_train)
    trainer.run(epochs = 1)
    #eval = trainer.evaluate(X_test, y_test)
    #print(f"precision: {eval['precision']}")
    trainer.save_down_model()
