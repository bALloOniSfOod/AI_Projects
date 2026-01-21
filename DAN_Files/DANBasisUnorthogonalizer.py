# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information

# This file constructs a Feature Similarity Matrix (FSM), which quantifies the correlations between features in a dataset, and
# applies this matrix to the dataset and any input vector, effectively converting it into a DAN network that is globally influenced
# by the data


import math
import copy
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix


def DANBasisUnorthogonalizer(theData, featureSimilarityAmplificationVectorExponentiation=1, featureSimilarityAmplificationMatrixExponentiation=1, inputVector=None, featureOutputs=True, dataOutputs=False, divideFeatureEntriesByDiagonal=True, returnFeatureMatrix=False, binarySolverBool=False):

    print("Preparing Data...")

    theData = copy.deepcopy(theData)

    outputBool = isinstance(theData[0][-1], list)
    if outputBool:
        DATAoutputHolder = [row[-1] for row in theData]
        theData = [row[:-1] for row in theData]

    X = np.asarray(theData, dtype=np.float64)

    print("Constructing Feature Similarity Matrix (FSM)...")

    if binarySolverBool:
        Xs = csr_matrix(X)
        FSM = (Xs.T @ Xs).toarray()
    else:
        FSM = X.T @ X

    if divideFeatureEntriesByDiagonal:
        maxDiagValue = np.max(np.diag(FSM))
        FSM = FSM / maxDiagValue

    FSM = FSM ** featureSimilarityAmplificationMatrixExponentiation

    print("Applying FSM to Data and Norming Result...")

    X_new = (X @ FSM.T) ** featureSimilarityAmplificationVectorExponentiation
    X_new = X_new / np.linalg.norm(X_new, axis=1, keepdims=True)

    if outputBool:
        print("Reattaching Original Outputs to Data...")
        X_new = X_new.tolist()
        for i in range(len(X_new)):
            X_new[i].append(DATAoutputHolder[i])
        X_new = np.asarray(X_new, dtype=object)

    if inputVector is None:
        if returnFeatureMatrix:
            print("Returning Output and FSM...")
            return X_new, FSM
        else:
            print("Returning Output...")
            return X_new

    print("Transforming Input Vector...")

    v = np.asarray(inputVector, dtype=np.float64)
    if v.shape[0] != FSM.shape[0]:
        raise ValueError("Mismatch between input vector length and feature count")

    v_new = (FSM @ v) ** featureSimilarityAmplificationVectorExponentiation
    v_norm = np.linalg.norm(v_new)
    if v_norm != 0:
        v_new /= v_norm

    print("Applying Input Vector to Unorthogonalized Data...")

    X_dot = np.asarray(X_new[:, :-1], dtype=np.float64) if outputBool else X_new
    X_dot = X_dot / np.linalg.norm(X_dot, axis=1, keepdims=True)

    similarity_scores = X_dot @ v_new

    if dataOutputs:
        print("Returning Data Outputs...")
        return similarity_scores.tolist()

    if featureOutputs:
        print("Returning Feature Outputs...")
        feature_scores = []
        X_base = np.asarray(theData, dtype=np.float64)
        for f in range(X_base.shape[1]):
            mask = X_base[:, f] != 0
            if np.any(mask):
                feature_scores.append(np.max(similarity_scores[mask]))
            else:
                feature_scores.append(0.0)
        return feature_scores





def DANInputUnorthogonalizer(inputVector, theData, normalizeInput=True, featureSimilarityAmplificationVectorExponentiation=1, featureSimilarityAmplificationMatrixExponentiation=1, divideFeatureEntriesByDiagonal=True, featureSimilarityMatrix=None):

    print("Unorthogonalizing Input Vector...")

    theData = copy.deepcopy(theData)

    if isinstance(theData[0][-1], list):
        theData = [row[:-1] for row in theData]

    X = np.asarray(theData, dtype=np.float64)

    if featureSimilarityMatrix is None:

        print("Constructing Feature Similarity Matrix...")

        FSM = X.T @ X

        if divideFeatureEntriesByDiagonal:
            maxDiagValue = np.max(np.diag(FSM))
            FSM = (FSM / maxDiagValue) ** featureSimilarityAmplificationMatrixExponentiation
        else:
            FSM = FSM ** featureSimilarityAmplificationMatrixExponentiation
    else:
        FSM = np.asarray(featureSimilarityMatrix, dtype=np.float64)

    v = np.asarray(inputVector, dtype=np.float64)
    if v.shape[0] != FSM.shape[0]:
        raise ValueError("Mismatch between input vector length and feature count")

    v_new = (FSM @ v) ** featureSimilarityAmplificationVectorExponentiation

    if normalizeInput:

        print("Normalizing Output...")

        norm = np.linalg.norm(v_new)
        if norm != 0:
            v_new /= norm

    print("Returning Transformed Vector...")

    return v_new.tolist()


if __name__ == "__main__":

    from Dataset import theData as theData1

    inputVec = DANInputUnorthogonalizer(theData1[0][:-1], theData=theData1, featureSimilarityAmplificationMatrixExponentiation=4)
    inputVec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    output = DANBasisUnorthogonalizer(theData1, featureSimilarityAmplificationMatrixExponentiation=4, inputVector=inputVec, featureOutputs=True, dataOutputs=False)
    print(output)





