# Written by Ryan Cerauli for the DAN Research Program headed by Anthony F. Beavers @ Indiana University. Copyright 2024. 
# See https://www.afbeavers.net/drg for more information


# This file constructs a working XOR gate utilizing a DAN


from DanClass import DAN

theData = [["Place1", "Place2", "Output"], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 0]]

myDAN = DAN(type="static", 
            excelDAN=False,
            pythonDAN=True,
            MAXSUBPython=False,
            orientation="horizontal",
            inputFormatting="clustered",
            newWorkbook="newHousingMarket.xlsx",   
            design=True, 
            ListOfLists=theData,
            originalWorkbook=None,
            dataSheet="Sheet1",
            categoryOrderPreservation=True,
            numericalAndAlphabeticalPreservation=True,
            allInputCategories=True,
            desiredModifications=[[]],
            categoryNames=True, 
            printStatements=True) 

myDAN.make()
myDAN.replaceInputsWith([["Place1", 1], ["Place2", 0]])
myDAN.showMaxValues()
print(myDAN.getBinaryOutput())
