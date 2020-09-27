import numpy as np
import pandas as pd

def getInfo(fileName):
    datasetDirectory = './dataset/'
    datasetFullName = datasetDirectory + fileName
    data = np.genfromtxt(datasetFullName, delimiter=',', dtype=None, encoding=None)
    #Now we are going to store the values in a keypair value
    #Note that we start at index 1 in order to avoid storing the headers into our dictionary
    infoDictionary = {}
    for i in range(1, len(data)):
        infoDictionary[int(data[i][0])] = (data[i][1])

    return infoDictionary

def getData(fileName):
    datasetDirectory = './dataset/'
    datasetFullName = datasetDirectory + fileName
    data = pd.read_csv(datasetFullName, header=None)
    return data


def writeResults(results, fileName):
    pd.DataFrame(results).to_csv('./results/' + fileName, header=None)
    print('Results Written to ' + fileName)


def generateGraph():
    print('Generate Graphs')
        

    