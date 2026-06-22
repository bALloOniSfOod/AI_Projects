from SimpleDANClass import simpleDAN
import numpy as np
import math
import copy
from copy import deepcopy
import random
from tqdm import tqdm



class ContinuousDAN:

    def __init__(self, continuousDatasetListOfLists, epsilon=100):
        self.continuousDatasetListOfLists = np.array(continuousDatasetListOfLists)
        self.epsilon = epsilon

        self.finalParameterizedListOfLists = []
        self.listOfColumnDicts = []

        for rowIndex in range(len(self.continuousDatasetListOfLists)):
            self.finalParameterizedListOfLists.append([])

        self.newFinalParameterizedListOfLists = copy.deepcopy(self.finalParameterizedListOfLists)

        for columnIndex in tqdm(range(len(self.continuousDatasetListOfLists[0]))):
            maxColumnVal = np.max(self.continuousDatasetListOfLists[:, columnIndex])
            minColumnVal = np.min(self.continuousDatasetListOfLists[:, columnIndex])
            columnBinSizes = (maxColumnVal - minColumnVal) / self.epsilon

            self.columnDict = {}

            for floatDictEntry in range(self.epsilon):
                self.columnDict[minColumnVal + (floatDictEntry * columnBinSizes)] = []

            self.listOfColumnDicts.append(self.columnDict)

            for rowIndex in range(len(self.continuousDatasetListOfLists)):
                rowColumnVal = self.continuousDatasetListOfLists[rowIndex, columnIndex]
                rowColumnValIndex = 0
                for floatDictEntry, keyVal in enumerate(self.columnDict.keys()):
                    if keyVal <= rowColumnVal and keyVal + columnBinSizes >= rowColumnVal:
                        rowColumnValIndex = floatDictEntry

                totalHolderList = []
                for floatDictEntry, keyVal in enumerate(self.columnDict.keys()):
                    finalHolderList = []
                    basinDif = self.epsilon - abs(floatDictEntry - rowColumnValIndex)
                    finalHolderList.extend([1,0] * basinDif)
                    finalHolderList.extend([0,1] * abs(floatDictEntry - rowColumnValIndex))
                    totalHolderList.append(finalHolderList)

                self.finalParameterizedListOfLists[rowIndex].append(totalHolderList)
        
        for rowIndex in tqdm(range(len(self.finalParameterizedListOfLists))):
            flattened_row = []
            for column_encoding in self.finalParameterizedListOfLists[rowIndex]:
                for bin_encoding in column_encoding:
                    flattened_row.extend(bin_encoding)
                    
            self.newFinalParameterizedListOfLists[rowIndex] = flattened_row

        self.continuousDAN = simpleDAN(self.newFinalParameterizedListOfLists)

    def parameterizeInput(self, inputList):
        inputList = np.array(inputList)
        
        finalParameterizedList = []
        
        for columnIndex in range(len(self.listOfColumnDicts)):
            columnDict = self.listOfColumnDicts[columnIndex]
            sortedKeys = sorted(columnDict.keys())
            
            if len(sortedKeys) > 1:
                columnBinSizes = sortedKeys[1] - sortedKeys[0]
            else:
                columnBinSizes = 1.0
                
            rowColumnVal = inputList[columnIndex]
            
            rowColumnValIndex = 0
            for floatDictEntry, keyVal in enumerate(sortedKeys):
                if keyVal <= rowColumnVal and keyVal + columnBinSizes >= rowColumnVal:
                    rowColumnValIndex = floatDictEntry
                    
            totalHolderList = []
            
            for floatDictEntry in range(self.epsilon):
                finalHolderList = []
                basinDif = self.epsilon - abs(floatDictEntry - rowColumnValIndex)
                
                finalHolderList.extend([1, 0] * basinDif)
                finalHolderList.extend([0, 1] * abs(floatDictEntry - rowColumnValIndex))
                totalHolderList.append(finalHolderList)
                
            finalParameterizedList.append(totalHolderList)
            
        flattenedInput = []
        for column_encoding in finalParameterizedList:
            for bin_encoding in column_encoding:
                flattenedInput.extend(bin_encoding)
                
        return flattenedInput
    
    def unparameterizeOutput(self, networkOutput):
        reconstructedInputs = []
        
        subspaceBitLength = 2 * self.epsilon
        
        columnBitLength = self.epsilon * subspaceBitLength
        
        for columnIndex in range(len(self.listOfColumnDicts)):
            columnDict = self.listOfColumnDicts[columnIndex]
            sortedKeys = sorted(columnDict.keys())
            
            minColumnVal = sortedKeys[0]
            if len(sortedKeys) > 1:
                columnBinSizes = sortedKeys[1] - sortedKeys[0]
            else:
                columnBinSizes = 1.0
                
            startColIdx = columnIndex * columnBitLength
            endColIdx = startColIdx + columnBitLength
            column_bits = networkOutput[startColIdx:endColIdx]
            
            winningBinIndex = 0
            maxOneCount = -1
            
            for binIndex in range(self.epsilon):
                startSubIdx = binIndex * subspaceBitLength
                endSubIdx = startSubIdx + subspaceBitLength
                subspaceBits = column_bits[startSubIdx:endSubIdx]
                
                oneCount = sum(1 for i in range(0, len(subspaceBits), 2) if subspaceBits[i] == 1)
                
                if oneCount > maxOneCount:
                    maxOneCount = oneCount
                    winningBinIndex = binIndex
            
            winningBinStart = minColumnVal + (winningBinIndex * columnBinSizes)
            winningValue = winningBinStart
            
            reconstructedInputs.append(winningValue)
            
        return reconstructedInputs

    def getOutput(self, inputList):
        parameterizedInputList = self.parameterizeInput(inputList)
        return self.unparameterizeOutput(self.continuousDAN.getOutput(parameterizedInputList))


if __name__ == "__main__":

    def generate_random_dataset(n_rows, m_cols, min_val=0.0, max_val=15.0):
        return [[random.uniform(min_val, max_val) for _ in range(m_cols)] for _ in range(n_rows)]

    dataset = generate_random_dataset(200, 10, min_val=0, max_val=11.0)    

    newData = [round(element, 0) for element in dataset[0]]
    otherNewData = [1,1,1,1,1,1,1,1,1,1]

    myContinuousDAN = ContinuousDAN(dataset)

    otherNewDataOutput = myContinuousDAN.getOutput(otherNewData)

    for i in range(len(otherNewDataOutput)):
        print(otherNewData[i], otherNewDataOutput[i])
