
# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This code enters an excel file with a dataset and converts it into a binary list of lists that is compatible
# with the other files in this repo


import pandas as pd
import xlwings as xw
# from DANANNMaker import DANtoANNNeuralNetGenerator
# from DANANNMaker import DANtoANNNeuralNetwork
import copy
from tqdm import tqdm
import openpyxl


import xlwings as xw


def ExcelDataToListofLists(originalWorkbook, dataSheet, ListOfListBool=True, CategoryListBool=True, desiredModifications=[[]]):

    print("Opening Excel File...")

    wk = xw.Book(originalWorkbook)
    ws = wk.sheets[dataSheet]

    print("Reading entire used range...")

    data = ws.used_range.value

    if not data or len(data) < 2:
        raise ValueError("Sheet appears to be empty or missing data")

    print("Collecting Data Categories...")
    CategoryList = data[0]
    DataMemberList = data[1:]

    print("Modifying Data...")

    if desiredModifications and desiredModifications[0]:
        for mod in desiredModifications:
            columnNum = mod[0] 
            action = mod[1]
            specificity = mod[2]

            if action == "round":
                for row in DataMemberList:
                    if row[columnNum] is not None:
                        row[columnNum] = round(row[columnNum], specificity)

            elif action == "splice":
                col_vals = [row[columnNum] for row in DataMemberList if row[columnNum] is not None]

                minVal = min(col_vals)
                maxVal = max(col_vals)

                spliceRange = (maxVal - minVal) / specificity
                spliceBins = [minVal + i * spliceRange for i in range(specificity + 1)]

                for row in DataMemberList:
                    val = row[columnNum]
                    for i in range(len(spliceBins) - 1):
                        if spliceBins[i] <= val <= spliceBins[i + 1]:
                            row[columnNum] = spliceBins[i + 1]
                            break

            else:
                raise ValueError(f"Unknown modification action: {action}")

    if ListOfListBool and CategoryListBool:
        return DataMemberList, CategoryList
    elif ListOfListBool:
        return DataMemberList
    elif CategoryListBool:
        return CategoryList
    else:
        raise ValueError("Must return data list and/or category list")




def ListofListsToBinaryEncodingListOfLists(ListOfLists, desiredOutputColumnList=None, includeOutputsInInputs=False, printBinaryDataset=False, binaryFinalOutputs=False, desiredModifications=[[]]):
    newBinaryEncodingListOfLists = []
    finalBinaryEncodingList = []
    finalCategoryElementList = []

    ListOfLists = copy.deepcopy(ListOfLists)

    print("Modifying Data...")

    if desiredModifications and desiredModifications[0]:
        for mod in desiredModifications:
            columnNum = mod[0] 
            action = mod[1]
            specificity = mod[2]

            if action == "round":
                for row in ListOfLists:
                    if row[columnNum] is not None:
                        row[columnNum] = round(row[columnNum], specificity)

            elif action == "splice":
                col_vals = [row[columnNum] for row in ListOfLists if row[columnNum] is not None]

                minVal = min(col_vals)
                maxVal = max(col_vals)

                spliceRange = (maxVal - minVal) / specificity
                spliceBins = [minVal + i * spliceRange for i in range(specificity + 1)]

                for row in ListOfLists:
                    val = row[columnNum]
                    for i in range(len(spliceBins) - 1):
                        if spliceBins[i] <= val <= spliceBins[i + 1]:
                            row[columnNum] = spliceBins[i + 1]
                            break

            else:
                raise ValueError(f"Unknown modification action: {action}")

    print("Constructing Binary Bins...")

    for ListOfListIndex in tqdm(range(len(ListOfLists[1]))):
        indexCategoryList = []
        for dataMember in ListOfLists:
            if [dataMember[ListOfListIndex], 0] not in indexCategoryList:
                indexCategoryList.append([dataMember[ListOfListIndex], 0])
        thisBinaryList = []
        thisCategoryList = []
        for dataList in indexCategoryList:
            thisBinaryList.append(dataList[1])
            thisCategoryList.append(dataList[0])
        finalBinaryEncodingList.append(thisBinaryList)
        finalCategoryElementList.append(thisCategoryList)

    print("Converting and Formatting Data to Binary...")

    for dataMember in tqdm(ListOfLists):
        binaryDataMemberListofLists = copy.deepcopy(finalBinaryEncodingList)
        for categoryIndex in range(len(dataMember)):
            for categoryElementIndex in range(len(finalCategoryElementList[categoryIndex])):
                if dataMember[categoryIndex] == finalCategoryElementList[categoryIndex][categoryElementIndex]:
                    binaryDataMemberListofLists[categoryIndex][categoryElementIndex] = 1
                    break
        finalOutputHolder = []
        if binaryFinalOutputs:
            for desiredOutputColumnListElement in desiredOutputColumnList:
                finalOutputHolder.append(binaryDataMemberListofLists[desiredOutputColumnListElement]) 
            finalFinalOutputHolder = []
            for elementList in finalOutputHolder:
                for listElement in elementList:
                    finalFinalOutputHolder.append(listElement)
            binaryDataMemberListofLists.append(finalFinalOutputHolder)
        else:
            for desiredOutputColumnListElement in desiredOutputColumnList:
                finalOutputHolder.append(dataMember[desiredOutputColumnListElement]) 
            binaryDataMemberListofLists.append(finalOutputHolder)
        if not includeOutputsInInputs:
            binaryDataMemberListofListsNew = [x for i, x in enumerate(binaryDataMemberListofLists) if i not in desiredOutputColumnList]
        else:
            binaryDataMemberListofListsNew = binaryDataMemberListofLists
        finalBinaryEncodingHolderList = []
        for categoryList in binaryDataMemberListofListsNew[:-1]:
            for categoryElement in categoryList:
                finalBinaryEncodingHolderList.append(categoryElement)
        finalBinaryEncodingHolderList.append(binaryDataMemberListofListsNew[-1])
        newBinaryEncodingListOfLists.append(finalBinaryEncodingHolderList)
    if printBinaryDataset:

        print("Printing Binary Dataset...")

        print(newBinaryEncodingListOfLists)

    print("Returning Binary Dataset")

    return newBinaryEncodingListOfLists
    
    



if __name__ == "__main__":

    outputList = []
    for i in range(998):
        outputList.append(i)

    
    OriginalListOfLists, categoryList = ExcelDataToListofLists("DataExcelWorkbook.xlsx", "Sheet2", desiredModifications=[[0, "splice", 150]])
    BinaryListOfLists = ListofListsToBinaryEncodingListOfLists(OriginalListOfLists, [0], printBinaryDataset=False, includeOutputsInInputs=True, binaryFinalOutputs=False)
    print("it was done")
    with open("/Users/bALloOniSfOod/Desktop/Achievements/AI-Chess-Project/Dataset.py", "w") as f:
        variable_name = "theData"
        f.write(f"{variable_name} = {BinaryListOfLists}\n")
        dict_name = "categoryDict"
        f.write(f"{dict_name} = {categoryList}")

    # DANNeuralNetHolder = DANtoANNNeuralNetGenerator(BinaryListOfLists)
    # DANNeuralNet = DANtoANNNeuralNetwork(DANNeuralNetHolder)
    # DANNeuralNet.getOutput([], True)
