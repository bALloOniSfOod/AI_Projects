
# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2026. 
# See https://www.afbeavers.net/drg for more information

# This file evaluates the accuracy of either of the MNIST files

from MNISTMHNClampedBinaryOutput01_60000TrainBeta100 import predictLabels, trueLabels

totalsum = 0
for i in range(len(predictLabels)):
    if predictLabels[i] == trueLabels[i]:
        totalsum += 1

print(totalsum, totalsum/len(predictLabels))


