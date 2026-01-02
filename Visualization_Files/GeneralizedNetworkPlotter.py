# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file visualizes the outputs of the networks in this repo in a variety of ways

import matplotlib.pyplot as plt
import math



def NetworkPlotCreator(outputDictOfLists, expectedOutputList, averageCommonOutputKeys=True, plotType="bar plot", designType="basic", plotTitle="Network Distance Plots", xAxisTitle="Networks", yAxisTitle="Distance From Expected Output"):

    print(f"Constructing {designType} {plotType} of {plotTitle}")

    if averageCommonOutputKeys: 
        xCategoryList = list(outputDictOfLists.keys())

        finalDistanceList = []
        for dictValue in list(outputDictOfLists.values()):
            dictValueDistanceList = []

            for valueList in dictValue:
                valueListDistanceList = 0

                for listElementIndex in range(len(valueList)):
                    valueListDistanceList += (valueList[listElementIndex] - expectedOutputList[listElementIndex]) ** 2
                
                dictValueDistanceList.append(math.sqrt(valueListDistanceList))
            
            finalDistanceList.append(sum(dictValueDistanceList) / len(dictValueDistanceList))

        if plotType == "bar plot":
            print(finalDistanceList)
            plt.bar(xCategoryList, finalDistanceList)
            plt.title(plotTitle)
            plt.xlabel(xAxisTitle)
            plt.ylabel(yAxisTitle)
            plt.show()

    if not averageCommonOutputKeys:
        xCategoryList = []

        networkOutputList = []
        keyList = list(outputDictOfLists.keys())
        i = 0
        for dictValue in list(outputDictOfLists.values()):
            j = 0
            for value in dictValue:
                xCategoryList.append(f"{keyList[i]}, {j}")
                networkOutputList.append(value)
                j += 1
            i += 1

        finalDistanceList = []
        for networkOutput in networkOutputList:
            valueDistanceList = 0
            
            for listIndex in range(len(networkOutput)):
                valueDistanceList += (networkOutput[listIndex] - expectedOutputList[listIndex]) ** 2

            finalDistanceList.append(math.sqrt(valueDistanceList))

        if plotType == "bar plot":
            plt.bar(xCategoryList, finalDistanceList)
            plt.title(plotTitle)
            plt.xlabel(xAxisTitle)
            plt.ylabel(yAxisTitle)
            plt.show()
