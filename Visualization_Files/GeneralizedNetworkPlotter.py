# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file visualizes the outputs of the networks in this repo in a variety of ways

import matplotlib.pyplot as plt
import math
import numpy as np


def NetworkPlotCreator(outputDictOfLists, expectedOutputList, averageCommonOutputKeys=True, plotType="bar plot", designType="basic", plotTitle="Network Distance Plots", xAxisTitle="Networks", yAxisTitle="Distance From Expected Output"):

    print(f"Constructing {designType} {plotType} of {plotTitle}")

    if plotType == "bar plot":

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

        
            plt.bar(xCategoryList, finalDistanceList)
            plt.title(plotTitle)
            plt.xlabel(xAxisTitle)
            plt.ylabel(yAxisTitle)
            plt.show()



    if plotType == "scatter plot trajectory":

        all_data = np.vstack([
            np.array(traj, dtype=float)
            for traj in outputDictOfLists.values()
        ])

        X_mean = all_data.mean(axis=0, keepdims=True)
        X_centered = all_data - X_mean

        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        PC = Vt[:2].T

        plt.figure(figsize=(8, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(outputDictOfLists)))

        for (label, traj), color in zip(outputDictOfLists.items(), colors):
            X = np.array(traj, dtype=float)
            Xp = (X - X_mean) @ PC

            plt.plot(Xp[:, 0], Xp[:, 1], color=color, linewidth=2, label=label)

            plt.scatter(Xp[:, 0], Xp[:, 1], color=color, s=30, alpha=0.8)

            plt.scatter(Xp[0, 0], Xp[0, 1], color=color, s=150, marker='o', edgecolors='black', linewidths=1.5, zorder=5)

            plt.scatter(Xp[-1, 0], Xp[-1, 1], color=color, s=150, marker='X', edgecolors='black', linewidths=1.5, zorder=5)

        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title("Distinct Output Trajectories (Start & End Points Marked)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
