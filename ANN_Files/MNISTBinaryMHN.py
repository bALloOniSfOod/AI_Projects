
# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2026. 
# See https://www.afbeavers.net/drg for more information

# This file uses the BinaryMHN class to train a binary MHN on the classic MINST dataset


from BinaryMHN import BinaryMHN
import h5py
import numpy as np
from scipy.io import loadmat
import json
from tqdm import tqdm
from ExcelDataToListofLists import ExcelDataToListofLists, ListofListsToBinaryEncodingListOfLists
print("begin")
from MNISTTrainData import theData, trainingLabels
print("import 1 complete")
from MNISTTestData import theData1, trainingLabels1
print("import 2 complete")

beta1 = 100

for cluster in tqdm(range(len(theData))):
    for element in range(len(theData[0])):
        if theData[cluster][element] > 0.01:
            theData[cluster][element] = 1
        else:
            theData[cluster][element] = 0

for cluster in tqdm(range(len(theData1))):
    for element in range(len(theData1[0])):
        if theData1[cluster][element] > 0.01:
            theData1[cluster][element] = 1
        else:
            theData1[cluster][element] = 0

combinedDataList = theData

for cluster in tqdm(theData1):
    theData.append(cluster)

encodedCombinedDataList = ListofListsToBinaryEncodingListOfLists(combinedDataList)

encodedTrainingListOfLists = encodedCombinedDataList[:60000]
encodedTestingListOfLists = encodedCombinedDataList[-10000:]

for i in range(len(encodedTrainingListOfLists)):
    holderOutput = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    holderOutput[trainingLabels[i]] = 1
    for j in range(len(holderOutput)):
        encodedTrainingListOfLists[i].append(holderOutput[j])

for i in range(len(encodedTestingListOfLists)):
    holderOutput = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(len(holderOutput)):
        encodedTestingListOfLists[i].append(holderOutput[j])

myMHN = BinaryMHN(encodedTrainingListOfLists, beta=beta1)

predictedOutputList = []

print("Doing things")

for test in tqdm(encodedTestingListOfLists):
    MHNOutput = myMHN.getOutputRobust(test, beta=beta1)[-10:]
    predictedOutputList.append(MHNOutput.index(max(MHNOutput)))

with open("MNISTMHNClampedBinaryOutput01_60000TrainBeta100.py", "w") as f:
    f.write("import json\n\n")
    f.write("trueLabels = ")
    json.dump(trainingLabels1, f)
    f.write("\n\npredictLabels = ")
    json.dump(predictedOutputList, f)
    f.write("\n")




