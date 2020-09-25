import utils
import decisionTree
import matplotlib.pyplot as plt
from sklearn import tree

def main():
    """ Driver.py """
    upperCaseLettersInfoFile = 'info_1.csv'
    greekLettersInfoFile = 'info_2.csv'
    trainingData1File = 'train_1.csv'
    trainingData2File = 'train_2.csv'

    val1File = 'val_1.csv'
    val2File = 'val_2.csv'

    testWithLabel1File = 'test_with_label_1.csv'
    testWithLabel2File = 'test_with_label_2.csv'

    baseDTFile1 = 'Base-DT-DS1.csv'
    baseDTFile2 = 'Base-DT-DS2.csv'

    bestDTFile1 = 'Best-DT-DS1.csv'
    bestDTFile2 = 'Best-DT-DS2.csv'


    upperCaseLettersDict = utils.getInfo(upperCaseLettersInfoFile)
    greekLettersInfo = utils.getInfo(greekLettersInfoFile)

    #Get data uses Pandas library. Returns 2d array with column headers in first row
    trainingData1 = utils.getData(trainingData1File)
    trainingData2 = utils.getData(trainingData2File)
    #Generating our ML Models
    baseDTUpperCase = decisionTree.generateBaseDT(trainingData1)
    baseDTGreek = decisionTree.generateBaseDT(trainingData2)
    bestDTUpperCase = decisionTree.generateBestDT(trainingData1)
    bestDTGreek = decisionTree.generateBestDT(trainingData2)
    
    val1Data = utils.getData(val1File)
    val2Data = utils.getData(val2File)
    testWithLabel1 = utils.getData(testWithLabel1File)
    testWithLabel2 = utils.getData(testWithLabel2File)

    # print('Running Validation for Base DT - Upper Case Letters...')
    # decisionTree.testModel(baseDTUpperCase, val1Data)
    # print('Running Validation for Best DT - Upper Case Letters...')
    # decisionTree.testModel(bestDTUpperCase, val1Data)

    # print('\nRunning Tests for Base DT - Upper Case Letters...')
    # baseDTRes1 = decisionTree.testModel(baseDTUpperCase, testWithLabel1)
    # utils.writeResults(baseDTRes1, baseDTFile1)
    # print('Running Tests for Best DT - Upper Case Letters...')
    # bestDTRes1 = decisionTree.testModel(bestDTUpperCase, testWithLabel1)
    # utils.writeResults(bestDTRes1, bestDTFile1)
    
    # print('\nRunning Validation for Base DT - Greek Letters...')
    # decisionTree.testModel(baseDTGreek, val2Data)
    # print('Running Validation for Best DT - Greek Letters...')
    # decisionTree.testModel(bestDTGreek, val2Data)
    
    # print('\nRunning Tests for Base DT - Greek Letters...')
    # baseDTRes2 = decisionTree.testModel(baseDTGreek, testWithLabel2)
    # utils.writeResults(baseDTRes2, baseDTFile2)
    # print('Running Tests for Best DT - Greek Letters...')
    # bestDTRes2 = decisionTree.testModel(bestDTGreek, testWithLabel2)
    # utils.writeResults(bestDTRes2, bestDTFile2)

    
    # utils.plotInstances(trainingData1, upperCaseLettersDict)
    

if __name__ == "__main__":
    main()