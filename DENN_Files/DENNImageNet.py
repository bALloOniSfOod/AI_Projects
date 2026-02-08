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
from DANBasisUnorthogonalizer import DANBasisUnorthogonalizer, DANInputUnorthogonalizer, AdjacentBinSimilarityCheck
import os
from tqdm import tqdm
from GeneralizedNetworkPlotter import NetworkPlotCreator
import random as rn
from JetsSharksANN import DataANNOutput, DataANNCreator
import math
import tensorflow

resolution = 25
rounding = 10
spliceNum = 3
FSMExp = 6

def DENNImageDatasetCreator(imagePathList, imageHeight=32, imageWidth=32, resolutionPrecision=1):

    print("Constructing Image Dataset in RGB (Non - Binary)...")

    dataset1 = []

    for imagePath in tqdm(imagePathList):
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
    


def CKAImageCompare(v1, v2):
    
    X = np.asarray(v1, dtype=np.float64)
    Y = np.asarray(v2, dtype=np.float64)

    assert X.ndim == 1 and Y.ndim == 1, "Inputs must be 1D vectors"
    assert X.shape == Y.shape, "Vectors must have equal length"

    X = X[:, None] 
    Y = Y[:, None] 

    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    XT_Y = X.T @ Y
    XT_X = X.T @ X
    YT_Y = Y.T @ Y

    print(f"CKA Image Similarity: ", float((np.linalg.norm(XT_Y, "fro") ** 2) / (np.linalg.norm(XT_X, "fro") * np.linalg.norm(YT_Y, "fro"))))


if __name__ == "__main__":

    DATASET_ROOT = "/Users/bALloOniSfOod/Downloads/mvtec_loco_anomaly_detection"

    numOfTraining = 310
    
    dataset = []
    for number in tqdm(range(330)):
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
    desiredModList = []
    for i in range(resolution * resolution * 3):
        desiredModList.append([i, "splice", spliceNum])
        outputList.append(i)

    dataset3 = ListofListsToBinaryEncodingListOfLists(dataset1, desiredOutputColumnList=outputList, includeOutputsInInputs=True, desiredModifications=desiredModList)

    test = dataset3[numOfTraining:]
    dataset2 = dataset3[:numOfTraining]
    for index in range(len(test)):
        test[index] = test[index][:-1]

    dataset = DANBasisUnorthogonalizer(dataset2, divideFeatureEntriesByDiagonal=True, returnModifiedFeatureMatrix=True, FSMMatrixExp=FSMExp, dataOutputs=True, featureOutputs=False)

    randomMatrix = [[rn.random() for j in range(len(dataset3[0][:-1]))] for i in range(numOfTraining)]
    for j in range(len(randomMatrix)):
        randomMatrix[j].append(dataset1[j][-1])

    dataset4 = []
    for index in range(len(dataset2)):
        dataset4.append(dataset2[index][:-1])


    # def customActivation(x, scale=15.0):
    #     x = torch.as_tensor(x, dtype=torch.float32)
    #     x = torch.clamp(x, -50, 50)
    #     x_div = x / scale
    #     return torch.sign(x) * torch.pow(10.0, torch.abs(x_div))

    def customFunction(param):
        return math.exp(param/600)
    
    def customFunction1(x):
        return tensorflow.math.exp(x/600.0)


    # uDENN Network #
    DANNeuralNetHolder = DANtoANNNeuralNetGenerator(dataset[0], conditionNumber=False, featureDANFirstLayerOutputBool=False, compressToANN=False, function=customFunction, leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0.000001, normalizeOutputs=False, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False)
    DANNeuralNet = DANtoANNNeuralNetwork(DANNeuralNetHolder, exportWeightMatrices=False, featureDANFirstLayerOutputBool=False)

    DANNeuralNetHolder3 = DANtoANNNeuralNetGenerator(dataset[0], conditionNumber=False, featureDANFirstLayerOutputBool=True, compressToANN=False, function=customFunction, leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0.000001, normalizeOutputs=False, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False)
    DANNeuralNet3 = DANtoANNNeuralNetwork(DANNeuralNetHolder3, exportWeightMatrices=False, featureDANFirstLayerOutputBool=True)

    # DAN Network #
    DANNeuralNetHolder1 = DANtoANNNeuralNetGenerator(dataset2, function=customFunction, featureDANFirstLayerOutputBool=False)
    DANNeuralNet1 = DANtoANNNeuralNetwork(DANNeuralNetHolder1, featureDANFirstLayerOutputBool=False)

    # Random Network #
    DANNeuralNetHolder2 = DANtoANNNeuralNetGenerator(randomMatrix, trainingData=dataset4, function=customFunction)
    DANNeuralNet2 = DANtoANNNeuralNetwork(DANNeuralNetHolder2)

    # Backprop Network #
    DataANNNetworkHolder = DataANNCreator(dataset2, activationFunction=customFunction1, epochs=500)

    for data in range(numOfTraining):
        
        inputVec = DANInputUnorthogonalizer(test[data], dataset3, True, featureSimilarityMatrix=dataset[1], FSMMatrixExp=FSMExp)
        
        output1 = DANNeuralNet.getOutput(inputVec)
        
        output2 = DANNeuralNet1.getOutput(test[data])

        output3 = DANNeuralNet2.getOutput(test[data])

        output5 = DANNeuralNet3.getOutput(inputVec)

        output4 = DataANNOutput([test[data]], DataANNNetworkHolder["W1"], DataANNNetworkHolder["W2"], activation=customFunction1)
        
        # for i in range(len(output1)):
        #     print(output1[i], output4[i], output2[i], output3[i], dataset1[data][i])
       
        NetworkPlotCreator({"uDENN": [output1], "Backprop": output4, "DAN": [output2], "Random": [output3], "Pool": [output5]}, expectedOutputList=dataset1[data])
        
        print("uDENN output:")
        DENNImageReconstructor(output1, resolution, resolution)
        print("DAN output:")
        DENNImageReconstructor(output2, resolution, resolution)
        print("Random output:")
        DENNImageReconstructor(output3, resolution, resolution)
        print("Pool output:")
        DENNImageReconstructor(output5, resolution, resolution)
        print("Backprop output:")
        DENNImageReconstructor(output4[0], resolution, resolution)
        print("Target output:")
        DENNImageReconstructor(dataset1[numOfTraining + data][:-1], resolution, resolution)


        print("uDENN, Backprop")
        CKAImageCompare(output1, output4[0])
        print("DAN, Backprop")
        CKAImageCompare(output2, output4[0])
        print("DAN, uDENN")
        CKAImageCompare(output1, output2)
        print("uDENN, Expected")
        CKAImageCompare(output1, dataset1[data][:-1])
        print("DAN, Expected")
        CKAImageCompare(output2, dataset1[data][:-1])
        print("Backprop, Expected")
        CKAImageCompare(output4[0], dataset1[data][:-1])


    for data in dataset2[0:10]:
        
        output = DANNeuralNet1.getOutput(data[:-1])
        
        DENNImageReconstructor(output, resolution, resolution)







