import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def generateBaseDT(trainingData):
    # print("Generating Base DT...")
    #We are getting every column except the last one (features)
    inputs = trainingData.iloc[:, :-1]
    #We are getting the last column (predicted value)
    values = trainingData.iloc[:, -1]
    clf = DecisionTreeClassifier(criterion='entropy')
    clf = clf.fit(inputs, values)
    # print('Base DT Generated and Trained\n')
    return clf

def generateBestDT(trainingData):
    inputs = trainingData.iloc[:, :-1]
    values = trainingData.iloc[:, -1]
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=5, min_impurity_decrease=0.0, class_weight='balanced')
    clf = clf.fit(inputs, values)
    return clf

