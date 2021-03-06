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
    greekLettersDict = utils.getInfo(greekLettersInfoFile)

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

    utils.plotInstances(trainingData1, upperCaseLettersDict, 'Uppercase Letters', 'Training', 'trainingUppercase')
    utils.plotInstances(trainingData2, greekLettersDict, 'Greek Letters', 'Training', 'trainingGreek')
    utils.plotInstances(val1Data, upperCaseLettersDict, 'Uppercase Letters', 'Validation', 'validationUppercase')
    utils.plotInstances(val2Data, greekLettersDict, 'Greek Letters', 'Validation', 'validationGreek')
    utils.plotInstances(testWithLabel1, upperCaseLettersDict, 'Uppercase Letters', 'Test', 'testUppercase')
    utils.plotInstances(testWithLabel2, greekLettersDict, 'Greek Letters', 'Test', 'testGreek')

    print('Running Validation for Base DT - Upper Case Letters...')
    utils.testModel(baseDTUpperCase, val1Data)
    print('Running Validation for Best DT - Upper Case Letters...')
    utils.testModel(bestDTUpperCase, val1Data)

    print('\nRunning Tests for Base DT - Upper Case Letters...')
    baseDTRes1 = utils.testModel(baseDTUpperCase, testWithLabel1)
    utils.writeMLResults(baseDTRes1, baseDTFile1)
    print('Running Tests for Best DT - Upper Case Letters...')
    bestDTRes1 = utils.testModel(bestDTUpperCase, testWithLabel1)
    utils.writeMLResults(bestDTRes1, bestDTFile1)
    
    print('\nRunning Validation for Base DT - Greek Letters...')
    utils.testModel(baseDTGreek, val2Data)
    print('Running Validation for Best DT - Greek Letters...')
    utils.testModel(bestDTGreek, val2Data)
    
    print('\nRunning Tests for Base DT - Greek Letters...')
    baseDTRes2 = utils.testModel(baseDTGreek, testWithLabel2)
    utils.writeMLResults(baseDTRes2, baseDTFile2)
    print('Running Tests for Best DT - Greek Letters...')
    bestDTRes2 = utils.testModel(bestDTGreek, testWithLabel2)
    utils.writeMLResults(bestDTRes2, bestDTFile2)

    utils.plotConfusionMatrix(baseDTUpperCase, testWithLabel1, upperCaseLettersDict, 'Base DT Uppercase Letters' ,'Base-DT-DS1-CM')
    utils.plotConfusionMatrix(baseDTGreek, testWithLabel2, greekLettersDict, 'Base DT Greek Letters' ,'Base-DT-DS2-CM')
    utils.plotConfusionMatrix(bestDTUpperCase, testWithLabel1, upperCaseLettersDict, 'Best DT Uppercase Letters' ,'Best-DT-DS1-CM')
    utils.plotConfusionMatrix(bestDTGreek, testWithLabel2, greekLettersDict, 'Best DT Greek Letters' ,'Best-DT-DS2-CM')

    utils.getClassificationReport(baseDTUpperCase, testWithLabel1, upperCaseLettersDict, 'Base DT Uppercase Letters' ,'Base-DT-DS1-Report')
    utils.getClassificationReport(baseDTGreek, testWithLabel2, greekLettersDict, 'Base DT Greek Letters' ,'Base-DT-DS2-Report')
    utils.getClassificationReport(bestDTUpperCase, testWithLabel1, upperCaseLettersDict, 'Base DT Uppercase Letters' ,'Best-DT-DS1-Report')
    utils.getClassificationReport(bestDTGreek, testWithLabel2, greekLettersDict, 'Base DT Greek Letters' ,'Best-DT-DS2-Report')


    

if __name__ == "__main__":
    main()