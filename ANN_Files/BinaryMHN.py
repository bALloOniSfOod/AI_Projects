
# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2026. 
# See https://www.afbeavers.net/drg for more information

# This file holds a BinaryMHN class that takes as input a binary list of lists and beta parameter and outputs a binary Modern Hopfield Network that can 
# can retrieve an output and get an energy


import numpy as np

class BinaryMHN:

    def __init__(self, binaryDataListOfLists, beta=1):
        self.binaryDataListOfLists = np.array(binaryDataListOfLists)
        self.beta = beta 

    def getOutput(self, binaryInputList, beta=None):
        b = beta if beta is not None else self.beta
        numpyBinaryInputList = np.array(binaryInputList)

        dot_product = self.binaryDataListOfLists @ numpyBinaryInputList
        initialOutput = np.exp(b * dot_product)

        weights = initialOutput / np.sum(initialOutput)
        scaled_matrix = self.binaryDataListOfLists * weights[:, np.newaxis]
        finalOutputNonBinary = np.sum(scaled_matrix, axis=0)

        return finalOutputNonBinary.tolist()
    
    def getOutputRobust(self, binaryInputList, beta=None):
        b = beta if beta is not None else self.beta
        numpyBinaryInputList = np.array(binaryInputList)

        exponent_powers = b * (self.binaryDataListOfLists @ numpyBinaryInputList)
        
        max_power = np.max(exponent_powers)
        initialOutput = np.exp(exponent_powers - max_power)

        weights = initialOutput / np.sum(initialOutput)

        scaled_matrix = self.binaryDataListOfLists * weights[:, np.newaxis]
        finalOutputNonBinary = np.sum(scaled_matrix, axis=0)

        return finalOutputNonBinary.tolist()
    
    def getEnergy(self, binaryInputList, beta=None):
        b = beta if beta is not None else self.beta
        numpyBinaryInputList = np.array(binaryInputList)

        dot_product = self.binaryDataListOfLists @ numpyBinaryInputList
        initialOutput = np.exp(b * dot_product)
        totalSum = np.sum(initialOutput)

        return -(1 / b) * np.log(totalSum)


if __name__ == "__main__":

    its = BinaryMHN([[1,0,1,0,0,1], [1,0,0,1,1,0], [0,1,1,0,1,0], [0,1,0,1,0,1]])
    print("\n")
    one = its.getOutput([1,0,1,0,0,0], beta=100)
    print(one)
    enegy = its.getEnergy([1,0,1,0,0,0], beta=100)
    print(enegy)

    




