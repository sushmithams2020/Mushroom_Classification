from LoadMushroomDataset import Load_mushroom_dataset,Load_MushroomDataset_with_Splits,Load_mushroom_dataset_RF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
import time as time
import pandas as pd
import numpy as np


class Mushroom():

    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def Random_Forest_classifier(self):
    
        rf_classifier = RandomForestClassifier(criterion='gini',n_estimators=10,random_state=10,max_features=10)

        start_time = time.time()
        # train the classifier
        rf_classifier.fit(self.x_train, self.y_train)
        # test the classifier
        y_predicted = rf_classifier.predict(self.x_test)
        # calculating accuracy
        accuracy = accuracy_score(self.y_test, y_predicted)
        f1score = f1_score(self.y_test, y_predicted)
        precisionscore = precision_score(self.y_test, y_predicted)
        recallscore = recall_score(self.y_test, y_predicted)

        # end time
        end_time = time.time()
        # calculate total time
        total_time = end_time - start_time

        # display accuracy and time
        print("-------------------------------------")
        print("Classifier: Random Forest")
        print("Accuracy:", (accuracy * 100), "%")
        print("F1 Score:", (f1score * 100), "%")
        print("Precision Score:", (precisionscore * 100), "%")
        print("Recall Score:", (recallscore * 100), "%")
        print("Time:", total_time, "sec")
        print("-------------------------------------")

        return accuracy

def ClassifyRF():

    x_train, x_test, y_train, y_test = Load_mushroom_dataset_RF()
    mushroom_model = Mushroom(x_train, x_test, y_train, y_test)
    accuracy = mushroom_model.Random_Forest_classifier()

    return accuracy

ClassifyRF()