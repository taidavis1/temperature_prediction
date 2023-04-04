import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
le = preprocessing.LabelEncoder()
scaler = preprocessing.MinMaxScaler()
class Data:
    def __init__(self , csv):
        self.csv = csv
    def read_data_csv(self):
        data = pd.read_csv(self.csv)
        return data
class Train_Data():
    def __init__(self , model , epochs , x_data , y_data , loss_func , optimizer):
        self.model = model
        self.epochs = epochs
        self.x_data = x_data
        self.y_data = y_data
        self.loss_func = loss_func
        self.optimizer = optimizer
    def train_data(self):
        # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_func,
            metrics=['accuracy']
        )
        history = self.model.fit(self.x_data, self.y_data, epochs=self.epochs)
    def overall_result(self , x_data_test , y_data_test):
        loss,accuracy = self.model.evaluate(x_data_test , y_data_test)
        print('Test loss:', loss)
        print('Test accuracy:', accuracy)
    def predictions(self, x_data_test):
        x_data_test = scaler.transform(x_data_test)
        predict = self.model.predict(x_data_test)
        return le.inverse_transform(predict.reshape(-1).round().astype(int))
#######################################################################################
if __name__ == '__main__':
    data = Data(csv= "t.csv")
    data = data.read_data_csv()
    x = data[['temp' , 'humidity' , 'windspeed' , 'cloudcover']].values
    y = data['conditions'].values
    y = le.fit_transform(y)
    x_train , x_test , y_train, y_test = train_test_split(x , y , shuffle=True , test_size=0.2)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_dim = 4 , activation='relu' , name= "input"),
        tf.keras.layers.Dense(5, activation = 'relu' , name = "hidden_layer1"),
        tf.keras.layers.Dense(10, activation = 'relu' , name = "hidden_layer2"),
        tf.keras.layers.Dense(15,name = "hidden_layer3"),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, name="output")
    ])
    train_d = Train_Data(model=model , epochs = 200 , x_data = x_train, y_data = y_train , loss_func = 'mse', optimizer = 'adam')
    train_d.train_data()
    print(train_d.predictions(np.array([[20 , 80 , 9 , 10]])))
    # print(train_d.predictions(x_test))
    train_d.overall_result(x_test , y_test)
    
