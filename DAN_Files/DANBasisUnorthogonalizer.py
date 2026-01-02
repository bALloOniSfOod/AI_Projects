# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file constructs a Feature Similarity Matrix (FSM), which quantifies the correlations between features in a dataset, and
# applies this matrix to the dataset and any input vector, effectively converting it into a DAN network that is globally influenced
# by the data


import math
from Dataset import theData as theData1
from tqdm import tqdm
from GeneralizedNetworkPlotter import NetworkPlotCreator
import copy




def DANBasisUnorthogonalizer(theData, normOutput=True, featureSimilarityAmplificationVectorExponentiation=1, featureSimilarityAmplificationMatrixExponentiation=1, inputVector=None, featureOutputs=False, dataOutputs=False, divideFeatureEntriesByDiagonal=True):

    print("Unorthogonalizing Basis...")

    theData = copy.deepcopy(theData)

    outputBool = False

    if type(theData[0][-1]) == list:
        outputBool = True
        DATAoutputHolder = []
        for listIndex in range(len(theData)):
            DATAoutputHolder.append(theData[listIndex][-1])
            theData[listIndex] = theData[listIndex][:-1]

    featureSimilarityMatrix = []

    print("Constructing Feature Similarity Matrix (FSM)...")

    maxDiagValue = 0
    for featureIndex in range(len(theData[0])):
        diagValue = 0
        for listIndex in range(len(theData)):
            if theData[listIndex][featureIndex] == 1:
                diagValue += 1
        if diagValue > maxDiagValue:
            maxDiagValue = diagValue

    featureSimilarityMatrix = []

    for primaryFeatureIndex in tqdm(range(len(theData[0]))):
        primaryFeatureList = []

        for secondaryFeatureIndex in range(len(theData[0])):
            featureSimilarityHolder = 0

            for listIndex in range(len(theData)):
                if (theData[listIndex][primaryFeatureIndex] == theData[listIndex][secondaryFeatureIndex] == 1):
                    featureSimilarityHolder += 1
            if divideFeatureEntriesByDiagonal:
                primaryFeatureList.append((featureSimilarityHolder / maxDiagValue) ** featureSimilarityAmplificationMatrixExponentiation)
            else:
                primaryFeatureList.append(featureSimilarityHolder ** featureSimilarityAmplificationMatrixExponentiation)

        featureSimilarityMatrix.append(primaryFeatureList)
    
    print("Applying Data to FSM...")

    theUnorthogonalizedData = []

    currentData = theData
    nextData = []

    for listIndex in tqdm(range(len(currentData))):
        newDataListHolder = []
        for listFeatureIndex in range(len(currentData[0])):
            dotProductHolder = 0
            for iterativeFeatureIndex in range(len(currentData[0])):
                dotProductHolder += currentData[listIndex][iterativeFeatureIndex] * featureSimilarityMatrix[listFeatureIndex][iterativeFeatureIndex]
            newDataListHolder.append(dotProductHolder ** featureSimilarityAmplificationVectorExponentiation)  ###################
        nextData.append(newDataListHolder)

        theUnorthogonalizedData = currentData

    theUnorthogonalizedData = nextData
    
    if normOutput:

        print("Normalizing Output...")

        for listIndex in tqdm(range(len(theUnorthogonalizedData))):
            normHolder = 0
            for featureIndex in range(len(theUnorthogonalizedData[0])):
                normHolder += (theUnorthogonalizedData[listIndex][featureIndex])**2
            normHolder = normHolder ** 0.5
            for featureIndex in range(len(theUnorthogonalizedData[0])):
                theUnorthogonalizedData[listIndex][featureIndex] = theUnorthogonalizedData[listIndex][featureIndex] / normHolder
    
    if outputBool:
        for listIndex in range(len(theData)):
            theUnorthogonalizedData[listIndex].append(DATAoutputHolder[listIndex])

    if inputVector:

        for listIndex in range(len(theUnorthogonalizedData)):
            theUnorthogonalizedData[listIndex] = theUnorthogonalizedData[listIndex][:-1]

        print("Transforming Input Vector...")

        newDataListHolder = []
        for listFeatureIndex in tqdm(range(len(theData[0]))):
            dotProductHolder = 0
            for iterativeFeatureIndex in range(len(theData[0])):
                try:
                    dotProductHolder += inputVector[iterativeFeatureIndex] * featureSimilarityMatrix[listFeatureIndex][iterativeFeatureIndex]
                except:
                    raise ValueError("Mismatch between input vector length and dataset feature length")
            newDataListHolder.append(dotProductHolder ** featureSimilarityAmplificationVectorExponentiation)  ###################
        transformedInputVector = newDataListHolder

        print("Normalizing Input Vector...")

        normHolder = 0
        for featureIndex in range(len(theUnorthogonalizedData[0])):
            normHolder += (transformedInputVector[featureIndex])**2
        normHolder = math.sqrt(normHolder)
        for featureIndex in range(len(theUnorthogonalizedData[0])):
            transformedInputVector[featureIndex] = transformedInputVector[featureIndex] / normHolder

        UnorthogonalizedDANOutputList = []

        print("Applying Input Vector to Unorthogonalized Data...")

        for listIndex in range(len(theUnorthogonalizedData)):
            dotProductHolder = 0
            for featureIndex in range(len(theUnorthogonalizedData[0])):
                dotProductHolder += theUnorthogonalizedData[listIndex][featureIndex] * transformedInputVector[featureIndex]
            UnorthogonalizedDANOutputList.append(dotProductHolder)

        if dataOutputs:

            print("Returning Data Similarity...")

            return UnorthogonalizedDANOutputList
        
        if featureOutputs:
            finalFeatureOutputList = []
            
            for featureIndex in tqdm(range(len(theUnorthogonalizedData[0]))):
                maxValHolder = 0
                for listIndex in range(len(theUnorthogonalizedData)):
                    if theData[listIndex][featureIndex] != 0 and UnorthogonalizedDANOutputList[listIndex] > maxValHolder:
                        maxValHolder = UnorthogonalizedDANOutputList[listIndex]
                
                finalFeatureOutputList.append(maxValHolder)

            print("Returning Feature Similarity...")

            return finalFeatureOutputList
    else:

        print("Returning Unorthogonalized Dataset...")

        return theUnorthogonalizedData




def DANInputUnorthogonalizer(inputVector, theData, normalizeInput=True, featureSimilarityAmplificationVectorExponentiation=1, featureSimilarityAmplificationMatrixExponentiation=1, divideFeatureEntriesByDiagonal=True):
    
    print("Unorthogonalizing Input Vector...")

    theData = copy.deepcopy(theData)

    if type(theData[0][-1]) == list:
        DATAoutputHolder = []
        for listIndex in range(len(theData)):
            DATAoutputHolder.append(theData[listIndex][-1])
            theData[listIndex] = theData[listIndex][:-1]

    print("Constructing Feature Similarity Matrix...")

    maxDiagValue = 0
    for featureIndex in range(len(theData[0])):
        diagValue = 0
        for listIndex in range(len(theData)):
            if theData[listIndex][featureIndex] == 1:
                diagValue += 1
        if diagValue > maxDiagValue:
            maxDiagValue = diagValue

    featureSimilarityMatrix = []

    for primaryFeatureIndex in tqdm(range(len(theData[0]))):
        primaryFeatureList = []

        for secondaryFeatureIndex in range(len(theData[0])):
            featureSimilarityHolder = 0

            for listIndex in range(len(theData)):
                if (theData[listIndex][primaryFeatureIndex] == theData[listIndex][secondaryFeatureIndex] == 1):
                    featureSimilarityHolder += 1
            if divideFeatureEntriesByDiagonal:
                primaryFeatureList.append((featureSimilarityHolder / maxDiagValue) ** featureSimilarityAmplificationMatrixExponentiation)
            else:
                primaryFeatureList.append(featureSimilarityHolder ** featureSimilarityAmplificationMatrixExponentiation)

        featureSimilarityMatrix.append(primaryFeatureList)

    newInputVector = []

    print("Applying Input Vector to Unorthogonalized Data...")

    for listFeatureIndex in tqdm(range(len(theData[0]))):
        dotProductHolder = 0
        for iterativeFeatureIndex in range(len(theData[0])):
            try:
                dotProductHolder += inputVector[iterativeFeatureIndex] * featureSimilarityMatrix[listFeatureIndex][iterativeFeatureIndex]
            except:
                raise ValueError("Mismatch between input vector length and dataset feature length")
        newInputVector.append(dotProductHolder ** featureSimilarityAmplificationVectorExponentiation)



    if normalizeInput:

        print("Normalizing Output...")

        normHolder = 0
        for featureIndex in range(len(newInputVector)):
            normHolder += (newInputVector[featureIndex])**2
        normHolder = normHolder ** 0.5
        for featureIndex in range(len(newInputVector)):
            newInputVector[featureIndex] = newInputVector[featureIndex] / normHolder

    print("Returning Transformed Vector...")

    return newInputVector


if __name__ == "__main__":

    output = DANBasisUnorthogonalizer(theData1, featureSimilarityAmplificationMatrixExponentiation=3, inputVector=theData1[0][:-1], featureOutputs=False, dataOutputs=True)
    print(output)



