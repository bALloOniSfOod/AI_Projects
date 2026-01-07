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

    
    dataset = []
    for number in tqdm(range(30)):
        if number < 10:
            num = f"00{number}"
        elif number < 100:
            num = f"0{number}"
        else:
            num = number
            
        img = os.path.join(DATASET_ROOT,"breakfast_box/train/good",f"{num}.png")

        dataset.append(img)

        print(f"image {number + 1} processed")
    
    dataset1 = DENNImageDatasetCreator(dataset, 16, 16)


    outputList = []
    for i in range(16 * 16 * 3):
        outputList.append(i)

    dataset1 = ListofListsToBinaryEncodingListOfLists(dataset1, desiredOutputColumnList=outputList, includeOutputsInInputs=True)

    test = dataset1[21:]
    dataset1 = dataset1[:21]
    for index in range(len(test)):
        test[index] = test[index][:-1]

    dataset = DANBasisUnorthogonalizer(dataset1, normOutput=True, divideFeatureEntriesByDiagonal=True, returnFeatureMatrix=True, featureSimilarityAmplificationMatrixExponentiation=4)



    DANNeuralNetHolder = DANtoANNNeuralNetGenerator(dataset[0], conditionNumber=False, compressToANN=False, function="exponential", leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0.000001, normalizeOutputs=False, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False)
    DANNeuralNet = DANtoANNNeuralNetwork(DANNeuralNetHolder, exportWeightMatrices=False)



    for data in test[1:10]:
        print(len(data))
        inputVec = DANInputUnorthogonalizer(data, dataset1, True, featureSimilarityMatrix=dataset[1], featureSimilarityAmplificationMatrixExponentiation=4)
        output = DANNeuralNet.getOutput(inputVec)
        
        DENNImageReconstructor(output, 16, 16)


    for data in dataset[0][1:10]:

        output = DANNeuralNet.getOutput(data[:-1])
        
        DENNImageReconstructor(output, 16, 16)




