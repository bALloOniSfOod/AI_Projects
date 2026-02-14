



import numpy as np
from DANBasisUnorthogonalizer import DANBasisUnorthogonalizer
import copy
import tqdm
from tqdm import tqdm
from JetsSharksDataHolder import jetsSharksMinusOutput

#### Hebbian-style kernal comparison ####

def neuronInspiredKernel(element1, element2, gammaParam=1):
    return (element1 * element2) / (np.abs(element1 - element2) + gammaParam)


#### Construct the layers ####

def dataTransformationEvolver(originalBinaryDataset, numOfLayers=3, divideLayer2ByDiagonal=True, kernelFunction=neuronInspiredKernel, gammaParam=1, transformationScalingFactor=1, maxEntries=1e8, pruneParam=0.1):


    hiddenLayer1ListOfLists = originalBinaryDataset
    hiddenLayer1ListOfListsCopy = copy.deepcopy(hiddenLayer1ListOfLists)

    finalNetworkDict = {}

    finalNetworkDict["1"] = hiddenLayer1ListOfLists

    if not divideLayer2ByDiagonal:
        hiddenLayer2 = DANBasisUnorthogonalizer(hiddenLayer1ListOfListsCopy, FSMMatrixExp=transformationScalingFactor, divideFeatureEntriesByDiagonal=False, returnOriginalFeatureMatrix=True, normalizeOutput=False)[1]
    else:
        hiddenLayer2 = DANBasisUnorthogonalizer(hiddenLayer1ListOfListsCopy, FSMMatrixExp=transformationScalingFactor, divideFeatureEntriesByDiagonal=True, returnModifiedFeatureMatrix=True, normalizeOutput=False)[1]

    layer2Dict = {}

    for rowIndexSlow in range(len(hiddenLayer2)):
        
        for columnIndexSlow in range(len(hiddenLayer2[0])):

            if hiddenLayer2[rowIndexSlow][columnIndexSlow] >= pruneParam:

                layer2Dict[f"{rowIndexSlow}x{columnIndexSlow}"] = hiddenLayer2[rowIndexSlow][columnIndexSlow]

    finalNetworkDict["2"] = layer2Dict
    
    previousLayerDict = layer2Dict

    for layerIndex in tqdm(range(numOfLayers - 2)):

        newLayerDict = {}

        for previousLayerKeySlow, previousLayerValueSlow in previousLayerDict.items():

            for previousLayerKeyFast, previousLayerValueFast in previousLayerDict.items():

                if kernelFunction(previousLayerValueSlow, previousLayerValueFast) >= pruneParam:

                    newLayerDict[f"{previousLayerKeySlow}x{previousLayerKeyFast}"] = kernelFunction(previousLayerValueSlow, previousLayerValueFast)
        
        finalNetworkDict[f"{layerIndex + 3}"] = newLayerDict

        previousLayerDict = newLayerDict

    return finalNetworkDict






def DIGNNEquationSolver(listOfNetworkListAndDicts, outputDict, DANFirstLayerBool=True, kernelFunction=neuronInspiredKernel):
    
    coefficientMatrixListOfLists = []
    i = 27
    for dataMemberList in listOfNetworkListAndDicts["1"]:

        print(i)
        if not DANFirstLayerBool:

            newFeatureList = dataMemberList


        if DANFirstLayerBool:

            newListOfNetworkListAndDicts = copy.deepcopy(listOfNetworkListAndDicts)

            newFeatureList = []

            initialOutputVector = []

            for rowIndex in range(len(listOfNetworkListAndDicts["1"])):

                dotProductSum = 0

                for columnIndex in range(len(listOfNetworkListAndDicts["1"][0])):

                    dotProductSum += (listOfNetworkListAndDicts["1"][rowIndex][columnIndex] * dataMemberList[columnIndex])

                initialOutputVector.append(dotProductSum)

            for rowIndex in range(len(listOfNetworkListAndDicts["1"])):

                for columnIndex in range(len(listOfNetworkListAndDicts["1"][0])): 

                    newListOfNetworkListAndDicts["1"][rowIndex][columnIndex] = listOfNetworkListAndDicts["1"][rowIndex][columnIndex] * initialOutputVector[rowIndex]
            
            for columnIndex in range(len(listOfNetworkListAndDicts["1"][0])):

                maxColumnVal = 0

                for rowIndex in range(len(listOfNetworkListAndDicts["1"])):

                    if newListOfNetworkListAndDicts["1"][rowIndex][columnIndex] > maxColumnVal:

                        maxColumnVal = newListOfNetworkListAndDicts["1"][rowIndex][columnIndex]

                newFeatureList.append(maxColumnVal)

        
        layerTwoInputTensorDict = {}

        for slowIndex in range(len(newFeatureList)):

            for fastIndex in range(len(newFeatureList)):

                if f"{slowIndex}x{fastIndex}" in listOfNetworkListAndDicts["2"].keys():

                    layerTwoInputTensorDict[f"{slowIndex}x{fastIndex}"] = kernelFunction(newFeatureList[slowIndex], newFeatureList[fastIndex])

        layerTwoOutputDict = {}

        for dictKeys in layerTwoInputTensorDict.keys():

            layerTwoOutputDict[dictKeys] = kernelFunction(layerTwoInputTensorDict[dictKeys], listOfNetworkListAndDicts["2"][dictKeys])

        

        newInputDict = layerTwoOutputDict

        for networkIndex in range(3, len(listOfNetworkListAndDicts) + 1):

            newOutputDict = {}

            for newInputKeySlow in newInputDict.keys():

                for newInputKeyFast in newInputDict.keys():

                    if f"{newInputKeySlow}x{newInputKeyFast}" in listOfNetworkListAndDicts[f"{networkIndex}"]:
                    
                        inputKernel = kernelFunction(newInputDict[f"{newInputKeySlow}"], newInputDict[f"{newInputKeyFast}"])

                        newOutputDict[f"{newInputKeySlow}x{newInputKeyFast}"] = kernelFunction(listOfNetworkListAndDicts[f"{networkIndex}"][f"{newInputKeySlow}x{newInputKeyFast}"], inputKernel)

            newInputDict = newOutputDict
        

        sortedKeys = sorted(newInputDict.keys()) 
        coefficientMatrixListOfLists.append([newInputDict[k] for k in sortedKeys])
        i -= 1
    return len(coefficientMatrixListOfLists[0])



        



if __name__ == "__main__":

    dataset = jetsSharksMinusOutput

    model = dataTransformationEvolver(dataset, numOfLayers=4, pruneParam=0.5)
    print(len(model["4"]), 41 ** 8)
    print(DIGNNEquationSolver(model, 0))
