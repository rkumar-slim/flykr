from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np

class Trainer(object):
    def __init__(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
        self.model = self.set_structure()

    def set_experiment_name(self):
        pass

    def set_structure(self):
        self.label_binarizer=LabelBinarizer()
        self.y_train = self.label_binarizer.fit_transform(self.y_train)
        self.y_val = self.label_binarizer.fit_transform(self.y_val)



        model=Sequential()
        model.add(Conv2D(128,kernel_size=(5,5),
                        strides=1,padding='same',activation='relu',input_shape=(100,100,1)))
        model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
        model.add(Conv2D(64,kernel_size=(2,2),
                        strides=1,activation='relu',padding='same'))
        model.add(MaxPool2D((2,2),2,padding='same'))
        model.add(Conv2D(32,kernel_size=(2,2),
                        strides=1,activation='relu',padding='same'))
        model.add(MaxPool2D((2,2),2,padding='same'))
        model.add(Flatten())

        model.add(Dense(units=512,activation='relu'))
        model.add(Dropout(rate=0.25))
        model.add(Dense(units=36,activation='softmax'))

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

        return model


    def run(self, epochs=1, batch_size=32):
        """Fit the model"""
        self.model.fit(self.X_train,
                       self.y_train,
                       validation_data=(self.X_val, self.y_val),
                       epochs = epochs, batch_size = batch_size)


    def evaluate(self, X_test, y_test):

        y_pred = self.model.predict(X_test)
        y_pred = (y_pred > 0.5) * 1.0

        #accuracy = accuracy_score()
        recall = recall_score(
            y_test,
            self.label_binarizer.inverse_transform(y_pred),
            labels=self.label_binarizer.classes_,
            average=None)
        precision = precision_score(
            y_test,
            self.label_binarizer.inverse_transform(y_pred),
            labels=self.label_binarizer.classes_,
            average=None)
        return {'recall': recall, 'precision': precision}

    def upload_model_to_gcp(self):
        pass

    def save_down_model(self):
        """Save the model into a .joblib format"""
        self.model.save('model.h5')
        print("Saving down model locally.")


if __name__ == "__main__":
    # Get and clean data
    data = pd.read_csv('./raw_data/asl_data_compressed_gray.csv')
    label = pd.read_csv('./raw_data/asl_data_label.csv')

    data = np.array(data).reshape(-1, 100, 100, 1)

    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        label,
                                                        test_size=0.1)
    # Train and save model, locally and
    trainer = Trainer(X_train, y_train)
    trainer.run()
    eval = trainer.evaluate(X_test, y_test)
    print(f"precision: {eval['precision']}")
    trainer.save_down_model()
