
# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information


# This is code for a compression algorithm that compresses a binary DAN into a non-binary DAN with the number of rows = rank(DAN). 
# Since a DAN is wholly constructed via the dataset, this amounts to essentially a PCA compression of the dataset followed by the 
# introduction of the DAN characteristics in order to gain its functionality as an AI network.


import numpy as np
from Dataset import theData
from Dataset import categoryDict as categoryDict
import copy


def compressDAN(dataset):

    print("Finding Independent Rows of Data...")

    outputs = []

    for index in range(len(dataset)):
        outputs.append(dataset[index][-1])
        dataset[index] = dataset[index][:-1]

    dataset = np.array(dataset, dtype=float)
    outputs = np.array(outputs, dtype=float)

    print("Constructing Row Basis...")

    independentRows = []
    for i in range(dataset.shape[0]):
        candidate = dataset[independentRows + [i], :] if independentRows else dataset[[i], :]
        if np.linalg.matrix_rank(candidate) > len(independentRows):
            independentRows.append(i)
    rowBasis = dataset[independentRows, :]
    r = len(rowBasis)

    print("Compressing Dataset...")
    
    Q, R = np.linalg.qr(rowBasis.T)
    compressedDataset = Q.T 

    print("Computing Coefficients for Reconstruction...")

    datasetCoefficients = np.zeros((dataset.shape[0], r))
    for j in range(dataset.shape[0]):
        datasetCoefficients[j] = np.linalg.lstsq(compressedDataset.T, dataset[j].reshape(-1,1), rcond=None)[0].flatten()

    outputBasis = outputs[independentRows]

    print("Returning Compressed Dataset, Reconstruction Coefficients, Output Basis, and Independent Rows...")

    return compressedDataset, datasetCoefficients, outputBasis, independentRows



def reconstructOutputs(InputVector, compressedDataset, datasetCoefficients, originalDataset, retrieveOutputVector=True, normOutput=True):
    
    numpyInputVector = np.array(InputVector, dtype=float)

    print("Projecting Input Vector to New Space...")

    yCompressed = compressedDataset @ numpyInputVector
    originalDatasetMinusOutput = copy.deepcopy(originalDataset)

    for index in range(len(originalDataset)):
        originalDatasetMinusOutput[index] = originalDatasetMinusOutput[index][:-1]

    if not retrieveOutputVector:
        return datasetCoefficients @ yCompressed
    
    else:

        print("Reconstructing Output in Original Basis...")

        theNewData = copy.deepcopy(originalDatasetMinusOutput)
        DANOutput = (datasetCoefficients @ yCompressed).tolist()
        for clusterIndex in range(len(originalDatasetMinusOutput)):
            for element in range(len(originalDatasetMinusOutput[clusterIndex])):
                theNewData[clusterIndex][element] = theNewData[clusterIndex][element] * DANOutput[clusterIndex]
        finalOutputVector = []
        for elementIndex in range(len(theNewData[0])):
            outputHolder = []
            for newClusterIndex in range(len(theNewData)):
                outputHolder.append(theNewData[newClusterIndex][elementIndex])
            maxVal = max(outputHolder)
            finalOutputVector.append(maxVal)
        if not normOutput:
            return finalOutputVector
        if normOutput:
            for i in range(len(finalOutputVector)):
                finalOutputVector[i] = finalOutputVector[i] / sum(InputVector)
            return finalOutputVector


if __name__ == "__main__":

    compressedDataset, datasetCoefficients, outputBasis, independentRows = compressDAN(theData)
    
    print(reconstructOutputs([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0], compressedDataset, datasetCoefficients, theData))

    

