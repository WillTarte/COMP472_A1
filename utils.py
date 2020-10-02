import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
import math

def plotInstances(trainingData, infoMapping, dataSet, dataSetType, fileName):
    """ dataset = Uppercase Letters or Greek Letters,
        datasetType = Training/Validation/Testing
    """
    print("Plotting")
    values = trainingData.iloc[:, -1]
    valueCounts = values.value_counts().to_dict()

    histoDict = {}

    for key, value in valueCounts.items():
        histoDict[infoMapping[key]] = value
    orderedHistoDict = collections.OrderedDict(sorted(histoDict.items()))
    names = list(orderedHistoDict.keys())
    values = list(orderedHistoDict.values())
    
    valuesSeries = pd.Series(np.array(values))

    plt.figure(figsize=(12, 8))
    colors = [
        '#2ebf44',
    ]
    ax = valuesSeries.plot(kind='bar', color=colors, edgecolor='black')
    ax.set_title('Class Instances of ' + dataSet + ' inside ' + dataSetType + ' data')
    ax.set_xlabel(dataSet)
    ax.set_ylabel('Occurences')
    ax.set_xticklabels(names)

    def add_value_labels(ax, spacing=5):
        """Add labels to the end of each bar in a bar chart.

        Arguments:
            ax (matplotlib.axes.Axes): The matplotlib object containing the axes
                of the plot to annotate.
            spacing (int): The distance between the labels and the bars.
        """

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label
            label = y_value

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va)                      # Vertically align label differently for
                                            # positive and negative values.


    # Call the function above. All the magic happens there.
    add_value_labels(ax)

    # plt.show()
    plt.savefig('./results/classInstances/' + fileName + '.png')

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


def writeMLResults(results, fileName):
    pd.DataFrame(results).to_csv('./results/mlResults/' + fileName, header=None)
    print('Results Written to ' + fileName)
