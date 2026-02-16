# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file preliminarily constructs a Tensor-Evolved Kernal Network (TEKN), which is a network that
# is grown via features, relationships between features, relationships between relationships of features,
# etc. for an arbitrary depth of the network (all the while pruning low-similarity nodes). An input is then 
# expanded via a similarity tensor and applied to the remaining nodes of a given layer, before that output is 
# similarily tensor-expanded and the process repeats, before finally undergoing a KRR-like operation to find the 
# final weights to fit a specific output. (KRR not implemented yet)



import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np
from DANBasisUnorthogonalizer import DANBasisUnorthogonalizer
import copy
import tqdm
from tqdm import tqdm
from JetsSharksDataHolder import theData
import random as rn


#### Hebbian-style kernal comparison ####


def neuronInspiredKernel(element1, element2, gammaParam=1):

    return (element1 * element2) / (np.abs(element1 - element2) + gammaParam)


#### Tensor-Evolved Kernel Network constructor ####


class TensorEvolvedKernelNetwork:
    
    def __init__(self, originalBinaryDataset, numOfLayers=3, divideLayer2ByDiagonal=True, DANFirstLayerBool=False, kernelFunction=neuronInspiredKernel, transformationScalingFactor=1, pruneParam=0.1, leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0.0001):

        originalBinaryDataset = copy.deepcopy(originalBinaryDataset)

        self.originalBinaryDataset = originalBinaryDataset
        self.numOfLayers = numOfLayers
        self.divideLayer2ByDiagonal = divideLayer2ByDiagonal
        self.DANFirstLayerBool = DANFirstLayerBool
        self.kernelFunction = kernelFunction
        self.transformationScalingFactor = transformationScalingFactor
        self.pruneParam = pruneParam
        self.leastSquareSolutionNorm = leastSquareSolutionNorm
        self.ridgeRegression = ridgeRegression
        self.lambdaVar = lambdaVar


        outputDict = {}


        print("Constructing output dictionary...")


        for columnIndex in range(len(originalBinaryDataset[0][-1])):

            outputVectorDictHolder = []

            for dataCluster in range(len(originalBinaryDataset)):

                outputVectorDictHolder.append(originalBinaryDataset[dataCluster][-1][columnIndex])

            outputDict[f"Output {columnIndex + 1}: "] = outputVectorDictHolder

        for rowIndex in range(len(originalBinaryDataset)):

            originalBinaryDataset[rowIndex] = originalBinaryDataset[rowIndex][:-1]


        print("Initializing layer 1...")    


        hiddenLayer1ListOfLists = originalBinaryDataset
        hiddenLayer1ListOfListsCopy = copy.deepcopy(hiddenLayer1ListOfLists)


        print("Initializing layer 2...")


        finalNetworkDict = {}

        finalNetworkDict["1"] = hiddenLayer1ListOfLists

        if not divideLayer2ByDiagonal:
            hiddenLayer2 = DANBasisUnorthogonalizer(hiddenLayer1ListOfListsCopy, FSMMatrixExp=transformationScalingFactor, divideFeatureEntriesByDiagonal=False, returnOriginalFeatureMatrix=True, normalizeOutput=False)[1]
        else:
            hiddenLayer2 = DANBasisUnorthogonalizer(hiddenLayer1ListOfListsCopy, FSMMatrixExp=transformationScalingFactor, divideFeatureEntriesByDiagonal=True, returnModifiedFeatureMatrix=True, normalizeOutput=False)[1]

        layer2Dict = {}

        for rowIndexSlow in tqdm(range(len(hiddenLayer2))):

            for columnIndexSlow in range(rowIndexSlow, len(hiddenLayer2[0])):

                value = hiddenLayer2[rowIndexSlow][columnIndexSlow]

                if value >= pruneParam:

                    layer2Dict[f"{rowIndexSlow}x{columnIndexSlow}"] = value


        finalNetworkDict["2"] = layer2Dict

        previousLayerDict = layer2Dict

        for layerIndex in tqdm(range(numOfLayers - 2)):

            print(f"Initializing layer {layerIndex + 2}...")

            newLayerDict = {}

            previousItems = list(previousLayerDict.items())

            for slowIndex in range(len(previousItems)):

                keySlow, valueSlow = previousItems[slowIndex]

                for fastIndex in range(slowIndex, len(previousItems)):

                    keyFast, valueFast = previousItems[fastIndex]

                    combinedKey = f"{keySlow}x{keyFast}"

                    kernelVal = kernelFunction(valueSlow, valueFast)

                    if kernelVal >= pruneParam:

                        newLayerDict[combinedKey] = kernelVal

            
            finalNetworkDict[f"{layerIndex + 3}"] = newLayerDict

            previousLayerDict = newLayerDict
        
        finalLayerKey = str(self.numOfLayers)

        self.FeatureKeyOrder = sorted(finalNetworkDict[finalLayerKey].keys())


        print("Training network...")


        DictOfNetworkListAndDicts = finalNetworkDict

        coefficientMatrixListOfLists = []
    
        for dataMemberList in tqdm(DictOfNetworkListAndDicts["1"]):

            if not DANFirstLayerBool:

                newFeatureList = dataMemberList

            if DANFirstLayerBool:

                newDictOfNetworkListAndDicts = copy.deepcopy(DictOfNetworkListAndDicts)

                newFeatureList = []

                initialOutputVector = []

                for rowIndex in range(len(DictOfNetworkListAndDicts["1"])):

                    dotProductSum = 0

                    for columnIndex in range(len(DictOfNetworkListAndDicts["1"][0])):
                        
                        dotProductSum += (DictOfNetworkListAndDicts["1"][rowIndex][columnIndex] * dataMemberList[columnIndex])

                    initialOutputVector.append(dotProductSum)

                for rowIndex in range(len(DictOfNetworkListAndDicts["1"])):

                    for columnIndex in range(len(DictOfNetworkListAndDicts["1"][0])): 

                        newDictOfNetworkListAndDicts["1"][rowIndex][columnIndex] = DictOfNetworkListAndDicts["1"][rowIndex][columnIndex] * initialOutputVector[rowIndex]
                
                for columnIndex in range(len(DictOfNetworkListAndDicts["1"][0])):

                    maxColumnVal = 0

                    for rowIndex in range(len(DictOfNetworkListAndDicts["1"])):

                        if newDictOfNetworkListAndDicts["1"][rowIndex][columnIndex] > maxColumnVal:

                            maxColumnVal = newDictOfNetworkListAndDicts["1"][rowIndex][columnIndex]

                    newFeatureList.append(maxColumnVal)



            layerTwoInputTensorDict = {}

            for slowIndex in range(len(newFeatureList)):

                for fastIndex in range(slowIndex, len(newFeatureList)):

                    combinedKey = f"{slowIndex}x{fastIndex}"

                    if combinedKey in DictOfNetworkListAndDicts["2"]:

                        layerTwoInputTensorDict[combinedKey] = kernelFunction(newFeatureList[slowIndex], newFeatureList[fastIndex])


            layerTwoOutputDict = {}

            for dictKeys in layerTwoInputTensorDict.keys():

                layerTwoOutputDict[dictKeys] = kernelFunction(layerTwoInputTensorDict[dictKeys], DictOfNetworkListAndDicts["2"][dictKeys])

            

            newInputDict = layerTwoOutputDict

            for networkIndex in range(3, self.numOfLayers + 1):

                newOutputDict = {}

                newInputItems = list(newInputDict.items())

                for slowIndex in range(len(newInputItems)):

                    newInputKeySlow, newInputValueSlow = newInputItems[slowIndex]

                    for fastIndex in range(slowIndex, len(newInputItems)):

                        newInputKeyFast, newInputValueFast = newInputItems[fastIndex]

                        combinedKey = f"{newInputKeySlow}x{newInputKeyFast}"

                        if combinedKey in DictOfNetworkListAndDicts[f"{networkIndex}"]:

                            inputKernel = kernelFunction(newInputValueSlow, newInputValueFast)

                            newOutputDict[combinedKey] = kernelFunction( DictOfNetworkListAndDicts[f"{networkIndex}"][combinedKey], inputKernel)


                newInputDict = newOutputDict
            

            featureVector = []

            for key in self.FeatureKeyOrder:

                featureVector.append(newInputDict.get(key, 0))

            coefficientMatrixListOfLists.append(featureVector)

            
        numpyCoefficientMatrix = np.array(coefficientMatrixListOfLists)


        print("Solving final system of equations...")


        self.SolutionsDict = {}

        for outputVecKey, outputVec in tqdm(outputDict.items()):

            numpyOutputVector = np.array(outputVec)

            if leastSquareSolutionNorm:
                Solutions = np.linalg.lstsq(numpyCoefficientMatrix, numpyOutputVector, rcond=None)[0]

            elif ridgeRegression:
                n = numpyCoefficientMatrix.shape[1]
                Solutions = np.linalg.solve(numpyCoefficientMatrix.T @ numpyCoefficientMatrix + self.lambdaVar * np.eye(n), numpyCoefficientMatrix.T @ numpyOutputVector)

            self.SolutionsDict[outputVecKey] = Solutions

        
        DictOfNetworkListAndDicts["Solutions"] = self.SolutionsDict

        self.DictOfNetworkListAndDicts = DictOfNetworkListAndDicts

        self.TEKN = DictOfNetworkListAndDicts
            
        print("Network trained!")



    def getOutput(self, inputVector):

        print("Getting output...")

        if not self.DANFirstLayerBool:

            newFeatureList = inputVector

        if self.DANFirstLayerBool:

            newDictOfNetworkListAndDicts = copy.deepcopy(self.DictOfNetworkListAndDicts)

            newFeatureList = []

            initialOutputVector = []

            for rowIndex in range(len(self.DictOfNetworkListAndDicts["1"])):

                dotProductSum = 0

                for columnIndex in range(len(self.DictOfNetworkListAndDicts["1"][0])):
                    
                    dotProductSum += (self.DictOfNetworkListAndDicts["1"][rowIndex][columnIndex] * inputVector[columnIndex])

                initialOutputVector.append(dotProductSum)

            for rowIndex in range(len(self.DictOfNetworkListAndDicts["1"])):

                for columnIndex in range(len(self.DictOfNetworkListAndDicts["1"][0])): 

                    newDictOfNetworkListAndDicts["1"][rowIndex][columnIndex] = self.DictOfNetworkListAndDicts["1"][rowIndex][columnIndex] * initialOutputVector[rowIndex]
            
            for columnIndex in range(len(self.DictOfNetworkListAndDicts["1"][0])):

                maxColumnVal = 0

                for rowIndex in range(len(self.DictOfNetworkListAndDicts["1"])):

                    if newDictOfNetworkListAndDicts["1"][rowIndex][columnIndex] > maxColumnVal:

                        maxColumnVal = newDictOfNetworkListAndDicts["1"][rowIndex][columnIndex]

                newFeatureList.append(maxColumnVal)



        layerTwoInputTensorDict = {}

        for slowIndex in range(len(newFeatureList)):

            for fastIndex in range(slowIndex, len(newFeatureList)):

                combinedKey = f"{slowIndex}x{fastIndex}"

                if combinedKey in self.DictOfNetworkListAndDicts["2"]:

                    layerTwoInputTensorDict[combinedKey] = self.kernelFunction(newFeatureList[slowIndex], newFeatureList[fastIndex])


        layerTwoOutputDict = {}

        for dictKeys in layerTwoInputTensorDict.keys():

            layerTwoOutputDict[dictKeys] = self.kernelFunction(layerTwoInputTensorDict[dictKeys], self.DictOfNetworkListAndDicts["2"][dictKeys])

        

        newInputDict = layerTwoOutputDict

        for networkIndex in range(3, self.numOfLayers + 1):

            newOutputDict = {}

            newInputItems = list(newInputDict.items())

            for slowIndex in range(len(newInputItems)):

                newInputKeySlow, newInputValueSlow = newInputItems[slowIndex]

                for fastIndex in range(slowIndex, len(newInputItems)):

                    newInputKeyFast, newInputValueFast = newInputItems[fastIndex]

                    combinedKey = f"{newInputKeySlow}x{newInputKeyFast}"

                    if combinedKey in self.DictOfNetworkListAndDicts[f"{networkIndex}"]:

                        inputKernel = self.kernelFunction(newInputValueSlow, newInputValueFast)

                        newOutputDict[combinedKey] = self.kernelFunction(self.DictOfNetworkListAndDicts[f"{networkIndex}"][combinedKey], inputKernel)


            newInputDict = newOutputDict
        

        inputDictAsVector = np.array([newInputDict.get(key, 0) for key in self.FeatureKeyOrder])


        outputs = {}

        for outputKey, solutionVector in self.SolutionsDict.items():
            outputs[outputKey] = np.dot(inputDictAsVector, solutionVector)

        return list(outputs.values())




            



if __name__ == "__main__":

    def randomJetsSharksInputList(partialActivation=True):

        nameList = [0] * 27
        nameList[rn.randrange(27)] = 1
        teamList = [0] * 2
        teamList[rn.randrange(2)] = 1
        ageList = [0] * 3
        ageList[rn.randrange(3)] = 1
        schoolList = [0] * 3
        schoolList[rn.randrange(3)] = 1
        marriedList = [0] * 3
        marriedList[rn.randrange(3)] = 1
        occList = [0] * 3
        occList[rn.randrange(3)] = 1

        finalList = [nameList, teamList, ageList, schoolList, marriedList, occList]

        if partialActivation:
            randNum = rn.randrange(6)
            for index in range(len(finalList[randNum])):
                finalList[randNum][index] = 0

        returnList = []

        for index1 in range(len(finalList)):
            for index2 in range(len(finalList[index1])):
                returnList.append(finalList[index1][index2])

        return returnList

    model = TensorEvolvedKernelNetwork(theData, numOfLayers=3, pruneParam=0.5, DANFirstLayerBool=False, ridgeRegression=True, leastSquareSolutionNorm=False)
 
    randomInput = randomJetsSharksInputList(True)

    dataInput = theData[0]

    # output = model.getOutput(dataInput)

    output = model.getOutput(randomInput)

    print(output)
