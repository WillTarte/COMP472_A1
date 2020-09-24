from numpy import genfromtxt

def getInfo(fileName):
    datasetDirectory = '../dataset/'
    datasetFullName = datasetDirectory + fileName
    data = genfromtxt(datasetFullName, delimiter=',', dtype=None, encoding=None)
    
    #Now we are going to store the values in a keypair value
    #Note that we start at index 1 in order to avoid storing the headers into our dictionary
    infoDictionary = {}
    for i in range(1, len(data)):
        infoDictionary[str(data[i][0])] = str(data[i][1])

    return infoDictionary

def getTestData(fileName):
    print("Getting Test Data")


def writeResults():
    print("Writing Results")


def generateGraph():
    print("Generate Graphs")
        

    