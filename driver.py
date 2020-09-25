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

    testWithLabel1File = 'test_with_label_1.csv'
    testWithLabel2File = 'test_with_label_2.csv'

    val1File = 'val_1.csv'
    val2File = 'val_2.csv'

    upperCaseLettersDict = utils.getInfo(upperCaseLettersInfoFile)
    greekLettersInfo = utils.getInfo(greekLettersInfoFile)
    # print('\nUppercase Letters Dict:')
    # print(upperCaseLettersDict)
    # print('\nGreek Letters Dict:')
    # print(greekLettersInfo)

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


    print('Running Validation for Base DT - Upper Case Letters...')
    decisionTree.testModel(baseDTUpperCase, val1Data)
    print('Running Validation for Best DT - Upper Case Letters...')
    decisionTree.testModel(bestDTUpperCase, val1Data)

    print('\nRunning Tests for Base DT - Upper Case Letters...')
    decisionTree.testModel(baseDTUpperCase, testWithLabel1)
    print('Running Tests for Best DT - Upper Case Letters...')
    decisionTree.testModel(bestDTUpperCase, testWithLabel1)
    
    print('\nRunning Validation for Base DT - Greek Letters...')
    decisionTree.testModel(baseDTGreek, val2Data)
    print('Running Validation for Best DT - Greek Letters...')
    decisionTree.testModel(bestDTGreek, val2Data)
    print('\nRunning Tests for Base DT - Greek Letters...')
    decisionTree.testModel(baseDTGreek, testWithLabel2)
    print('Running Tests for Best DT - Greek Letters...')
    decisionTree.testModel(bestDTGreek, testWithLabel2)
    

if __name__ == "__main__":
    main()