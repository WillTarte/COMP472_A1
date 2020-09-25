import utils
import baseDT
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

    upperCaseLettersDict = utils.getInfo(upperCaseLettersInfoFile)
    greekLettersInfo = utils.getInfo(greekLettersInfoFile)
    print('\nUppercase Letters Dict:')
    print(upperCaseLettersDict)
    print('\nGreek Letters Dict:')
    print(greekLettersInfo)

    #Get data uses Pandas library. Returns 2d array with column headers in first row
    trainingData1 = utils.getData(trainingData1File)
    trainingData2 = utils.getData(trainingData2File)
    #Generating our ML Models
    baseDTUpperCase = baseDT.generateBaseDT(trainingData1)
    baseDTGreek = baseDT.generateBaseDT(trainingData2)
    #We dont need to do use validation dataset on default. Because we don't need to tune parameters
    testWithLabel1 = utils.getData(testWithLabel1File)
    testWithLabel2 = utils.getData(testWithLabel2File)
    
    baseDT.testModel(baseDTUpperCase, testWithLabel1)
    baseDT.testModel(baseDTGreek, testWithLabel2)
    

if __name__ == "__main__":
    main()