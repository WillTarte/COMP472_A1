from typing import List
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

def getData(fname: str) -> (List[List[float]], List[float]):

    with open("./dataset/" + fname, "r") as f:
        X = []
        Y = []
        for line in f:
            line_tokens = line.split(",")
            X.append(list(map(float, line_tokens[:-1])))
            Y.append(float(line_tokens[-1]))
    
    return (X, Y)

if __name__ == "__main__":

    print("Loading training datasets.")

    ds1_training_X, ds1_training_Y = getData("train_1.csv")
    ds2_training_X, ds2_training_Y = getData("train_2.csv")

    print("Loading validation datatsets.")
    ds1_val_X, ds1_val_Y = getData("val_1.csv")
    ds2_val_X, ds2_val_Y = getData("val_2.csv")

    print("Creating base MLP classifiers and fitting to training set.")

    mlpClassifier_1 = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="sgd")
    mlpClassifier_2 = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver="sgd")

    mlpClassifier_1.fit(ds1_training_X, ds1_training_Y)
    mlpClassifier_2.fit(ds2_training_X, ds2_training_Y)

    print("Predicting classes of validation datasets.")
    ds1_out = mlpClassifier_1.predict(ds1_val_X)
    ds2_out = mlpClassifier_2.predict(ds2_val_X)

    print("Done. Outputting report.")
    print(classification_report(ds1_val_Y, ds1_out))
    print(classification_report(ds2_val_Y, ds2_out))


    print("Using Grid Search for hyperparameter tuning")
    grid_search_values = {"activation": ["identity", "logistic", "tanh", "relu"], "hidden_layer_sizes": [(30,50), (10,10,10)], "solver": ["sgd", "adam"]}

    gridsearch_mlp_1 = GridSearchCV(MLPClassifier(), param_grid=grid_search_values)
    gridsearch_mlp_2 = GridSearchCV(MLPClassifier(), param_grid=grid_search_values)

    gridsearch_mlp_1.fit(ds1_training_X, ds1_training_Y)
    gridsearch_mlp_2.fit(ds2_training_X, ds2_training_Y)

    grid_ds1_out = gridsearch_mlp_1.predict(ds1_val_X)
    grid_ds2_out = gridsearch_mlp_1.predict(ds2_val_X)

    print("Done. Outputting report.")
    print(classification_report(ds1_val_Y, grid_ds1_out))
    print(classification_report(ds2_val_Y, grid_ds2_out))
