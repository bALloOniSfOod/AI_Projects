
# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information


# This code constructs a DENN. More information about these networks can be found in About_This_File.txt. It takes as input a dataset and 
# constructs a DENN. 


import numpy as np
from tqdm import tqdm
import math
from Dataset import theData as theData
from Dataset import categoryDict
from Dataset import theLabeledData
from JetsSharksDataHolder import theAutoEncoderData
import ANNMatrices
import copy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import gmres
from DANBasisUnorthogonalizer import DANBasisUnorthogonalizer, DANInputUnorthogonalizer
from GeneralizedNetworkPlotter import NetworkPlotCreator
import random as rn
from JetsSharksANN import DataANNCreator, DataANNOutput
from DanClass import DAN
import tensorflow




################################################
### Construct Neural Network Equation Solver ###
################################################

def NNEquationSolver(DataMemberList, trainingData=None, function="exponential", conditionNumber=False, compressToANN=False, leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0, normalizeOutputs=True, maxAlignment=True, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False, solveExpectedEntropy=False):
    
    ######################################################################
    ### Construct Output Vector and Data Member List Excluding Outputs ###

    outputVectorDict = {}
    print("Formatting Data...")
    for column in range(len(DataMemberList[-1][-1])):
        outputVectorDictHolder = []
        for dataCluster in range(len(DataMemberList)):
            outputVectorDictHolder.append(DataMemberList[dataCluster][-1][column])
        outputVectorDict[f"Output {column + 1}: "] = outputVectorDictHolder
    DataListMinusOutput = []

    if compressToANN:
        print("Compressing Dataset...")
        if linearCompression:
            originalDataMemberList = copy.deepcopy(DataMemberList)
            otherOriginalDataMemberList = copy.deepcopy(DataMemberList)
            for item in originalDataMemberList:
                item.pop(-1)
            for item in otherOriginalDataMemberList:
                item.pop(-1)  
            Ab = np.array(otherOriginalDataMemberList, dtype=float)
            A = Ab[:, :]
            for key in outputVectorDict.keys():
                outputVectorDict[key] = np.array(outputVectorDict[key], dtype=float)
            r = np.linalg.matrix_rank(A)
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            row_basis = Vt[:r, :]
            coeffs = np.linalg.lstsq(row_basis.T, A.T, rcond=None)[0].T
            compressed_rows = []
            for i in range(r):
                row = np.zeros(A.shape[1])
                for j in range(A.shape[0]):
                    row += coeffs[j, i] * A[j]
                compressed_rows.append(row)
            compressed_rows = np.array(compressed_rows)
            for key in outputVectorDict.keys():
                outputVectorDict[key] = coeffs.T @ outputVectorDict[key]
            DataMemberList = compressed_rows.tolist()
            for key in outputVectorDict.keys():
                outputVectorDict[key] = outputVectorDict[key].tolist()
            for item in DataMemberList:
                DataListMinusOutput.append(item)
    

    else:
        for dataCluster in DataMemberList:
            newCluster = dataCluster[:-1]
            DataListMinusOutput.append(newCluster)
    
    if solveExpectedEntropy:
        print("Solving Expected Entropy...")
        totalSum = 0
        for memberIndex in tqdm(range(len(DataListMinusOutput))):
            for elementIndex in range(len(DataListMinusOutput[memberIndex])):
                totalSum += DataListMinusOutput[memberIndex][elementIndex]
        entropyList = []
        for inputClusterIndex in tqdm(range(len(DataListMinusOutput))):
            totalDot = 0
            for iterateClusterIndex in range(len(DataListMinusOutput)):
                for elementIndex in range(len(DataListMinusOutput[0])):
                    totalDot += DataListMinusOutput[inputClusterIndex][elementIndex] * DataListMinusOutput[iterateClusterIndex][elementIndex]
            clusterEntropy = (totalDot / totalSum) * math.log2(totalSum / totalDot)
            entropyList.append(clusterEntropy)

    #######################################
    ### Constructing Coefficient Matrix ###

    #######################################
    ### Constructing Coefficient Matrix ###
    #######################################

    print("Constructing Coefficient Matrix...")

    firstLayerMatrix = DataListMinusOutput

    coefficientMatrix = []

    for inputDataCluster in tqdm(trainingData if trainingData is not None else firstLayerMatrix):
        finalEquation = []
        for iterativeDataCluster in firstLayerMatrix:
            dotProductSum = sum(inputDataCluster[i] * iterativeDataCluster[i] for i in range(len(iterativeDataCluster)))

            if function == "":
                finalEquation.append(dotProductSum)
            elif function == "exponential":
                finalEquation.append(10 ** (dotProductSum/len(iterativeDataCluster)) if dotProductSum > 0 else -10 **(dotProductSum/len(iterativeDataCluster)))
            elif function == "sigmoid":
                finalEquation.append(1 / (1 + math.exp(-dotProductSum/len(inputDataCluster))))
            elif function == "tanh":
                finalEquation.append(math.tanh(dotProductSum/len(inputDataCluster)))
            elif function == "relu":
                finalEquation.append(max(0, dotProductSum/len(inputDataCluster)))

        coefficientMatrix.append(finalEquation)




    ##############################################################
    ### Construct Numpy Arrays for Solving System of Equations ###

    print("Constructing Numpy Matrix...")
    numpyCoefficientMatrix = np.array(coefficientMatrix)


    if conditionNumber:
        print("Finding Shape, Rank, Condition Number...")
        K = numpyCoefficientMatrix
        print("K shape:", K.shape)
        rank_K = np.linalg.matrix_rank(K)
        print("rank(K):", rank_K)
        try:
            condK = np.linalg.cond(K)
        except Exception:
            condK = float('inf')
        print("cond(K):", condK)

    #################################
    ### Solve System of Equations ###

    SolutionsDict = {}
    print("Training Outputs...")
    for outputVecKey, outputVec in tqdm(outputVectorDict.items()):
        numpyOutputVector = np.array(outputVec)
        if leastSquareSolutionNorm:
            Solutions = np.linalg.lstsq(numpyCoefficientMatrix, numpyOutputVector, rcond=None)
        elif fastBinaryEquationSolver:
            Solutions = gmres(numpyCoefficientMatrix, numpyOutputVector, restart=50)
        elif ridgeRegression:
            n = numpyCoefficientMatrix.shape[1]
            Solutions = [np.linalg.solve(numpyCoefficientMatrix.T @ numpyCoefficientMatrix + lambdaVar * np.eye(n), numpyCoefficientMatrix.T @ numpyOutputVector)]
        SolutionsDict[outputVecKey] = Solutions[0]
        
        

    #############################################################################
    ### Define 1st Layer Matrix and 2nd Layer Matrix for the Class ###


    firstLayerMatrix = DataMemberList
    secondLayerMatrixDict = SolutionsDict
    

    print("Number of Hidden Nodes: ", len(DataMemberList))

    ######################################
    ### Return Neural Network Matrices ###

    return firstLayerMatrix, secondLayerMatrixDict, function


################################
### Construct DAN->ANN Class ###
################################

class DANtoANNNeuralNetGenerator:

    #################################
    ### Initialize Neural Network ###
    #################################

    def __init__(self, DataMemberList, trainingData=None, firstLayer=[], secondLayerDict={}, function="", compressToANN=False, leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0, conditionNumber=False, normalizeOutputs=True, maxAlignment=True, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False):
        if firstLayer and secondLayerDict:
            self.firstLayerMatrix = firstLayer
            self.secondLayerMatrix = secondLayerDict
            self.function = function
        else:
            self.DataMemberList = DataMemberList
            self.function = function
            if compressToANN:
                if leastSquareSolutionNorm:
                    self.firstLayerMatrix, self.secondLayerMatrix, self.function = NNEquationSolver(DataMemberList=self.DataMemberList, function=self.function, trainingData=trainingData, compressToANN=True, conditionNumber=conditionNumber, normalizeOutputs=normalizeOutputs, maxAlignment=maxAlignment, linearCompression=linearCompression, nonLinearCompression=nonLinearCompression, fastBinaryEquationSolver=fastBinaryEquationSolver)
                else:
                    self.firstLayerMatrix, self.secondLayerMatrix, self.function = NNEquationSolver(DataMemberList=self.DataMemberList, function=self.function, trainingData=trainingData, compressToANN=True, leastSquareSolutionNorm=leastSquareSolutionNorm, ridgeRegression=ridgeRegression, lambdaVar=lambdaVar, conditionNumber=conditionNumber, normalizeOutputs=normalizeOutputs, maxAlignment=maxAlignment, linearCompression=linearCompression, nonLinearCompression=nonLinearCompression, fastBinaryEquationSolver=fastBinaryEquationSolver)
            else:
                if leastSquareSolutionNorm:
                    self.firstLayerMatrix, self.secondLayerMatrix, self.function = NNEquationSolver(DataMemberList=self.DataMemberList, function=self.function, trainingData=trainingData, compressToANN=False, conditionNumber=conditionNumber, normalizeOutputs=normalizeOutputs, maxAlignment=maxAlignment, linearCompression=linearCompression, nonLinearCompression=nonLinearCompression, fastBinaryEquationSolver=fastBinaryEquationSolver)
                else:
                    self.firstLayerMatrix, self.secondLayerMatrix, self.function = NNEquationSolver(DataMemberList=self.DataMemberList, function=self.function, trainingData=trainingData, compressToANN=False, leastSquareSolutionNorm=leastSquareSolutionNorm, ridgeRegression=ridgeRegression, lambdaVar=lambdaVar, conditionNumber=conditionNumber, normalizeOutputs=normalizeOutputs, maxAlignment=maxAlignment, linearCompression=linearCompression, nonLinearCompression=nonLinearCompression, fastBinaryEquationSolver=fastBinaryEquationSolver)

    #######################
    ### Add Data Member ###
    #######################

    def addData(self, newDataMember):
        self.DataMemberList.append(newDataMember)
        self.firstLayerMatrix, self.secondLayerMatrix, self.function = NNEquationSolver(self.DataMemberList, self.function)

    #############################
    ### Take Away Data Member ###
    #############################

    def removeData(self, toBeRemovedDataMember):
        for dataMember in self.DataMemberList:
            if dataMember == toBeRemovedDataMember:
                self.DataMemberList.remove(toBeRemovedDataMember)
                break
        self.firstLayerMatrix, self.secondLayerMatrix, self.function = NNEquationSolver(self.DataMemberList, self.function)


######################################################
### Construct DAN->ANN Neural Network Object Class ###
######################################################

class DANtoANNNeuralNetwork:
    def __init__(self, NeuralNetGeneratorObject, exportWeightMatrices=False):
        self.firstLayerMatrix = NeuralNetGeneratorObject.firstLayerMatrix
        self.secondLayerMatrix = NeuralNetGeneratorObject.secondLayerMatrix
        self.function = NeuralNetGeneratorObject.function
        self.exportWeightMatrices = exportWeightMatrices

        if exportWeightMatrices:
            with open("/Users/bALloOniSfOod/Desktop/Achievements/AI-Chess-Project/ANNMatrices.py", "w") as f:
                f.write("matrixList = [\n")
                f.write("    [\n")
                for row in self.firstLayerMatrix:
                    f.write(f"        {row},\n")
                f.write("    ],\n")
                # f.write("    [\n")
                # for val in listSecondLayerMatrix:
                #     f.write(f"        {val},\n")
                # f.write("    ]\n")

                # f.write("]\n")


    ###################################################
    ### Run Inputs through First Layer and Function ###

    def getOutput(self, inputDataCluster, printWeights=False, outputsAsList=True):
        print("Getting Output...")
        if self.exportWeightMatrices:
            firstLayer = ANNMatrices.matrixList[0]
            secondLayer = ANNMatrices.matrixList[1]
        else:
            firstLayer = self.firstLayerMatrix
            secondLayer = self.secondLayerMatrix

        firstLayerOutput = []
        if printWeights:
            print("Printing Weights...")
            print(secondLayer)
        for iterativeDataCluster in firstLayer:
            dotProductSum = 0
            for index in range(len(inputDataCluster)):
                dotProductSum += inputDataCluster[index] * iterativeDataCluster[index]
            ### insert function "if" statements here ###
            if self.function == "":
                firstLayerOutput.append(dotProductSum)
            elif self.function == "exponential":
                firstLayerOutput.append(10 ** (dotProductSum/len(iterativeDataCluster)) if dotProductSum > 0 else -10 **(dotProductSum/len(iterativeDataCluster)))
            elif self.function == "sigmoid":
                val = 1 / (1 + math.exp(-(dotProductSum/len(inputDataCluster))))
                firstLayerOutput.append(val)
            elif self.function == "tanh":
                val = math.tanh(dotProductSum/len(inputDataCluster))
                firstLayerOutput.append(val)
            elif self.function == "relu":
                val = max(0, dotProductSum/len(inputDataCluster))
                firstLayerOutput.append(val)

        #################################################
        ### Run Function Outputs through Second Layer ###

        OutputDict = {}
        for key, secondLayer in self.secondLayerMatrix.items():
            totalSum = 0
            for index in range(len(firstLayerOutput)):
                totalSum += firstLayerOutput[index] * secondLayer[index]
            OutputDict[key] = round(float(totalSum), 6)

        #####################
        ### Return Output ###

        if outputsAsList:
            return list(OutputDict.values())
        
        if not outputsAsList:
            return OutputDict





if __name__ == "__main__":

    dataset1 = DANBasisUnorthogonalizer(theData, featureSimilarityAmplificationMatrixExponentiation=2)
    dataset = theData
    datasetMinusOutput = copy.deepcopy(theData)
    for index in range(len(datasetMinusOutput)):
        datasetMinusOutput[index] = datasetMinusOutput[index][:-1]


    DANNeuralNetHolder = DANtoANNNeuralNetGenerator(dataset, conditionNumber=True, compressToANN=False, function="exponential", leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0.00001, normalizeOutputs=False, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False)
    DANNeuralNet = DANtoANNNeuralNetwork(DANNeuralNetHolder, exportWeightMatrices=False)

    DANNeuralNetHolder1 = DANtoANNNeuralNetGenerator(dataset1, conditionNumber=True, compressToANN=False, function="exponential", leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=0.00001, normalizeOutputs=False, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False)
    DANNeuralNet1 = DANtoANNNeuralNetwork(DANNeuralNetHolder1, exportWeightMatrices=False)


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


    inputVecList = []
    for i in range(100):
        inputVecList.append(randomJetsSharksInputList())


    theLabeledData.insert(0, categoryDict)


    newDAN = DAN(type="static", excelDAN=False, pythonDAN=True, MAXSUBPython=False, ListOfLists=theLabeledData)
    newDAN.make()


    def customActivation(x):
        x_div = x / 6 
        return tensorflow.where(x > 0, 10.0 ** x_div, -10.0 ** x_div)

    JSANNOutput = DataANNCreator(theData, epochs=150, activationFunction='relu')

    for inputVec in inputVecList:
        # inputVec = dataset[0][:-1]
        inputVec1 = DANInputUnorthogonalizer(inputVec, theData, featureSimilarityAmplificationMatrixExponentiation=2)
    
        orthogonalizedDANOutput = DANNeuralNet.getOutput(inputVec)
        unorthogonalizedDANOutput = DANNeuralNet1.getOutput(inputVec1)


        NetworkDict = {"DAN": [orthogonalizedDANOutput], "uDENN": [unorthogonalizedDANOutput]}

        randomNetworkListofLists = []
        rows = 27
        cols = 41
        for i in range(5):
            matrix = [[rn.random() for j in range(cols)] for i in range(rows)]
            for j in range(len(matrix)):
                matrix[j].append(theData[j][-1])
            DANNeuralNetHolder2 = DANtoANNNeuralNetGenerator(matrix, trainingData=datasetMinusOutput, conditionNumber=True, compressToANN=False, function="", leastSquareSolutionNorm=True, ridgeRegression=False, lambdaVar=1, normalizeOutputs=False, linearCompression=True, nonLinearCompression=False, fastBinaryEquationSolver=False)
            DANNeuralNet2 = DANtoANNNeuralNetwork(DANNeuralNetHolder2, exportWeightMatrices=False)
            unorthogDANOutput = DANNeuralNet2.getOutput(inputVec)
            randomNetworkListofLists.append(unorthogDANOutput)

        NetworkDict["Random"] = randomNetworkListofLists

        dataListMinusOutput = []
        outputVector = []
        for dataCluster in theData:
            newCluster = dataCluster[:-1]
            output = dataCluster[-1]
            dataListMinusOutput.append(newCluster)
            outputVector.append(output) 

        A = np.array(dataListMinusOutput, dtype=float)

        b = np.array(outputVector, dtype=float) 

        x = np.array(inputVec, dtype=float)
        
        newNewData = copy.deepcopy(dataListMinusOutput)
        DANOutput = []
        otherx = x.tolist()
        for cluster in range(len(newNewData)):
            dot = 0
            for element in range(len(newNewData[cluster])):
                dot += otherx[element] * newNewData[cluster][element]
            DANOutput.append(dot)
        for clusterIndex in range(len(newNewData)):
            for element in range(len(newNewData[clusterIndex])):
                newNewData[clusterIndex][element] = newNewData[clusterIndex][element] * DANOutput[clusterIndex]
        finalOutputVector = []
        for elementIndex in range(len(newNewData[0])):
            outputHolder = []
            for newClusterIndex in range(len(newNewData)):
                outputHolder.append(newNewData[newClusterIndex][elementIndex])
            maxVal = max(outputHolder)
            finalOutputVector.append(maxVal)
        b_original = finalOutputVector

        for index in range(len(b_original)):
            b_original[index] = b_original[index]/6

        # b_original = newDAN.getMaxValues()

        NetworkDict["Expected"] = [b_original]

        # DENNOutput = DANBasisUnorthogonalizer(theData, True, featureSimilarityAmplificationMatrixExponentiation=1, inputVector=inputVec)

        # NetworkDict["DENN"] = [DENNOutput]

        # print(DENNOutput)

        theJSANNOutput = DataANNOutput([inputVec], JSANNOutput["W1"], JSANNOutput["W2"], activation='relu')

        # paramDict, trajects = JetsSharksANNCreator(theData=theData, epochs=500, returnOutputTrajectories=True, inputVector=inputVec)
        print(theJSANNOutput)
        NetworkDict["Backprop"] = theJSANNOutput
        
        NetworkPlotCreator(NetworkDict, b_original)
        
        # trajects = JetsSharksANNCreator(
        # theData=theData,
        # epochs=500,
        # returnOutputTrajectories=True,
        # inputVector=inputVec
        # )

        # otherNetworkDict = {"backprop": trajects}

        # NetworkPlotCreator(otherNetworkDict, unorthogDANOutput)

        # for i in range(len(JSANNOutput)):
        #     print(math.sqrt((unorthogDANOutput[i] - JSANNOutput[0][i]) ** 2))

        # print(dataset[0][-1])
        # print(unorthogonalizedDANOutput)
        # print(orthogonalizedDANOutput)
        # print(randomNetworkListofLists[0], "\n")
        # print(dataset[0][-1])
        # print(NetworkDict)

        # paramDict, trajects = JetsSharksANNCreator(theData=theData, epochs=500, returnOutputTrajectories=True, inputVector=inputVec)

        # NetworkDict["Backprop"] = trajects

        # NetworkPlotCreator(NetworkDict, b_original, plotType="scatter plot trajectory")



    

    

    



    

    


    
