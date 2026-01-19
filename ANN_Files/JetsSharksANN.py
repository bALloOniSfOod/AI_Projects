# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file initializes and trains an artificial neural network on the Jets and Sharks dataset


import tensorflow
from tensorflow.keras import layers, models
import numpy as np


def customActivation(x):

    if not isinstance(x, (tensorflow.Tensor, tensorflow.Variable)):
        x = tensorflow.convert_to_tensor(x, dtype=tensorflow.float32)

    x_div = x / 15.0
    positive = tensorflow.pow(10.0, x_div)
    negative = -tensorflow.pow(10.0, x_div)

    return tensorflow.where(x > 0, positive, negative)



def DataANNCreator( theData, activationFunction='tanh', epochs=500, useBias=False, returnOutputTrajectories=False, inputVector=None):

    output_trajectory = []

    class OutputTracker(tensorflow.keras.callbacks.Callback):
        def __init__(self, input_vector):
            super().__init__()
            self.input_vector = np.array(input_vector, dtype=float).reshape(1, -1)

        def on_epoch_end(self, epoch, logs=None):
            y = self.model.predict(self.input_vector, verbose=0)
            output_trajectory.append(y.flatten().tolist())

    input_dim = len(theData[0]) - 1

    sample_output = theData[0][-1]
    if isinstance(sample_output, (list, tuple, np.ndarray)):
        output_dim = len(sample_output)
    else:
        output_dim = 1


    if activationFunction in [None, "linear"]:
        activation = None
    else:
        activation = activationFunction

    model = models.Sequential([layers.Dense( len(theData), activation=activation, input_shape=(input_dim,), use_bias=useBias), layers.Dense(output_dim, use_bias=useBias)])

    model.compile(optimizer='adam', loss='mse')


    x_train = np.array([row[:-1] for row in theData], dtype=float)

    if output_dim == 1:
        y_train = np.array([[row[-1]] for row in theData], dtype=float)
    else:
        y_train = np.array([row[-1] for row in theData], dtype=float)

    callbacks = []
    if returnOutputTrajectories and inputVector is not None:
        callbacks.append(OutputTracker(inputVector))


    model.fit( x_train, y_train, epochs=epochs, batch_size=32, verbose=0, callbacks=callbacks)

    
    params = {}

    if useBias:
        W1, b1 = model.layers[0].get_weights()
        W2, b2 = model.layers[1].get_weights()
        params["W1"] = W1
        params["b1"] = b1
        params["W2"] = W2
        params["b2"] = b2
    else:
        params["W1"] = model.layers[0].get_weights()[0]
        params["W2"] = model.layers[1].get_weights()[0]

    if returnOutputTrajectories:
        return params, output_trajectory
    else:
        return params





def DataANNOutput(inputVectorListOfLists, weight1, weight2, bias1=None, bias2=None, activation="relu"):

    finalOutputList = []

    def apply_activation(z):
        if callable(activation):
            out = activation(z)
            if hasattr(out, "numpy"):
                out = out.numpy()
            return out
        elif activation in [None, "linear"]:
            return z
        elif activation == "relu":
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    input_dim = weight1.shape[0] 

    for inputVector in inputVectorListOfLists:

        X = np.array(inputVector[:input_dim], dtype=float).reshape(1, -1)

        Z1 = X @ weight1
        if bias1 is not None:
            Z1 = Z1 + bias1
        A1 = apply_activation(Z1)

        Z2 = A1 @ weight2
        if bias2 is not None:
            Z2 = Z2 + bias2

        finalOutputList.append(Z2.flatten().tolist())

    return finalOutputList




# with open("/Users/bALloOniSfOod/Desktop/Achievements/AI-Chess-Project/JetsSharksANNOutput.py", "w") as f:
#         variable_name = "JSANNOutput"
#         f.write(f"{variable_name} = {outputHolder}\n")
        
        
