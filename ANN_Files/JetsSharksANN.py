# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file initializes and trains an artificial neural network on the Jets and Sharks dataset


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.datasets import mnist
from Dataset import theData
import numpy as np

input_dim = len(theData[0][:-1])
outputHolder = []

for i in range(10):
    model = keras.Sequential([
        layers.Dense(len(theData), activation='relu', input_shape=(input_dim,)),
        layers.Dense(input_dim)
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    x_trainList = []
    y_trainList = []

    inputList1 = []

    for i in range(len(theData)):
        theData[i][input_dim] = [round(v, 2) for v in theData[i][input_dim]]


    for dataMember in theData:
        x_trainList.append(dataMember[:input_dim])  
        inputList1.append(dataMember[:input_dim])
        y_trainList.append(dataMember[:input_dim])

    inputList = np.array(inputList1)

    x_train = np.array([row[:input_dim] for row in theData], dtype=float)
    y_train = np.array([row[input_dim] for row in theData], dtype=float) 

    print(y_train.tolist())
    model.fit(x_train, y_train, epochs=400, batch_size=32, validation_split=0.1)

    predictions = model.predict(inputList)
    outputHolder.append(list(predictions[0]))
    # for index in range(len(predictions)):
    #     print(list(predictions[index]))
    #     print(y_trainList[index])
    #     print("\n")


with open("/Users/bALloOniSfOod/Desktop/Achievements/AI-Chess-Project/JetsSharksANNOutput.py", "w") as f:
        variable_name = "JSANNOutput"
        f.write(f"{variable_name} = {outputHolder}\n")
        
