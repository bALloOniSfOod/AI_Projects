# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This code constructs functionalities to make image networks utilizing DENNs and uDENNs, and plots
# results of these networks


from PIL import Image
import torchvision.transforms as T
import torch
import numpy as np
from DANANNMaker import DANtoANNNeuralNetGenerator, DANtoANNNeuralNetwork
import copy
from copy import deepcopy
from torchvision.transforms import ToPILImage
import ExcelDataToListofLists
from ExcelDataToListofLists import ExcelDataToListofLists, ListofListsToBinaryEncodingListOfLists
from DANBasisUnorthogonalizer import DANBasisUnorthogonalizer, DANInputUnorthogonalizer
import os
from tqdm import tqdm
from GeneralizedNetworkPlotter import NetworkPlotCreator
import random as rn
from JetsSharksANN import DataANNOutput, DataANNCreator
import tensorflow

resolution = 15
rounding = 3

def DENNImageDatasetCreator(imagePathList, imageHeight=32, imageWidth=32, resolutionPrecision=1):

    print("Constructing Image Dataset in RGB (Non - Binary)...")

    dataset1 = []

    for imagePath in imagePathList:
        img = Image.open(imagePath).convert("RGB")
        transform = T.Compose([T.Resize((imageHeight, imageWidth)), T.ToTensor()])
        tensor = transform(img)
        vec = tensor.flatten()
        vec = vec.tolist()
        vec = [round(x, resolutionPrecision) for x in vec]
        vec1 = copy.deepcopy(vec)
        vec.append(vec1)

        dataset1.append(vec)

    return dataset1
    



def DENNImageReconstructor(imageVector, imageHeight=32, imageWidth=32):
    
    print("Reconstructing Image Vector...")

    tensor = torch.tensor(imageVector)
    tensor = tensor.reshape(3, imageHeight, imageWidth)
    img = ToPILImage()(tensor)
    img.show()
    



if __name__ == "__main__":

    DATASET_ROOT = "/Users/bALloOniSfOod/Downloads/mvtec_loco_anomaly_detection"

    numOfTraining = 45
    
    dataset = []
    for number in tqdm(range(349)):
        if number < 10:
            num = f"00{number}"
        elif number < 100:
            num = f"0{number}"
        else:
            num = number
            
        img = os.path.join(DATASET_ROOT,"breakfast_box/train/good",f"{num}.png")

        dataset.append(img)

        print(f"image {number + 1} processed")
    
    dataset1 = DENNImageDatasetCreator(dataset, resolution, resolution, rounding)


    outputList = []
    for i in range(resolution * resolution * 3):
        outputList.append(i)

    dataset3 = ListofListsToBinaryEncodingListOfLists(dataset1, desiredOutputColumnList=outputList, includeOutputsInInputs=True)

    test = dataset3[numOfTraining:]
    dataset2 = dataset3[:numOfTraining]
    for index in range(len(test)):
        test[index] = test[index][:-1]

    dataset = DANBasisUnorthogonalizer(dataset2, normOutput=True, divideFeatureEntriesByDiagonal=True, returnFeatureMatrix=True, featureSimilarityAmplificationMatrixExponentiation=3)

    randomMatrix = [[rn.random() for j in range(len(dataset3[0][:-1]))] for i in range(numOfTraining)]
    for j in range(len(randomMatrix)):
        randomMatrix[j].append(dataset1[j][-1])

    dataset4 = []
    for index in range(len(dataset2)):
        dataset4.append(dataset2[index][:-1])


    def customActivation(x, scale=15.0):
        x = tensorflow.convert_to_tensor(x, dtype=tensorflow.float32)
        x_div = x / scale

        return tensorflow.sign(x) * tensorflow.pow(10.0, tensorflow.abs(x_div))


    # uDENN Network #
    DANNeuralNetHolder = DANtoANNNeuralNetGenerator(dataset[0], conditionNumber=False, compressToANN=False, function="exponential", leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0.000001, normalizeOutputs=False, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False)
    DANNeuralNet = DANtoANNNeuralNetwork(DANNeuralNetHolder, exportWeightMatrices=False)

    # DAN Network #
    DANNeuralNetHolder1 = DANtoANNNeuralNetGenerator(dataset2, function="exponential")
    DANNeuralNet1 = DANtoANNNeuralNetwork(DANNeuralNetHolder1)

    # Random Network #
    DANNeuralNetHolder2 = DANtoANNNeuralNetGenerator(randomMatrix, trainingData=dataset4, function="exponential")
    DANNeuralNet2 = DANtoANNNeuralNetwork(DANNeuralNetHolder2)

    # Backprop Network #
    DataANNNetworkHolder = DataANNCreator(dataset2, activationFunction=customActivation)

    for data in range(numOfTraining):
        
        inputVec = DANInputUnorthogonalizer(test[data], dataset3, True, featureSimilarityMatrix=dataset[1], featureSimilarityAmplificationMatrixExponentiation=3)
        
        output1 = DANNeuralNet.getOutput(inputVec)
        
        output2 = DANNeuralNet1.getOutput(test[data])

        output3 = DANNeuralNet2.getOutput(test[data])

        output4 = DataANNOutput([test[data]], DataANNNetworkHolder["W1"], DataANNNetworkHolder["W2"])
        
        for i in range(len(output1)):
            print(output1[i], output2[i], output3[i], output4[0][i], dataset1[data][i])
       
        NetworkPlotCreator({"uDENN": [output1], "DAN": [output2], "Random": [output3], "Backprop": output4}, expectedOutputList=dataset1[data])
        
        print("uDENN output:")
        DENNImageReconstructor(output1, resolution, resolution)
        print("DAN output:")
        DENNImageReconstructor(output2, resolution, resolution)
        print("Random output:")
        DENNImageReconstructor(output3, resolution, resolution)
        print("Backprop output:")
        DENNImageReconstructor(output4, resolution, resolution)
        print("Target output:")
        DENNImageReconstructor(dataset1[data + 304][:-1], resolution, resolution)


    for data in dataset2[0:10]:
        
        output = DANNeuralNet1.getOutput(data[:-1])
        
        DENNImageReconstructor(output, resolution, resolution)



