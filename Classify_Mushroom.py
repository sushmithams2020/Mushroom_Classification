from LoadMushroomDataset import Load_mushroom_dataset,Load_MushroomDataset_with_Splits
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time as time

class Mushroom():

    def __init__(self,x_data,y_data,x_data_pca):
        self.x_data = x_data
        self.y_data = y_data
        self.x_data_pca = x_data_pca

    def classifiers(self):

        classifiers={'Naive Bayes':BernoulliNB(),'Logistic Regression':LogisticRegression(solver='newton-cg')}

        accuracies = []
        for key,val_classifier in classifiers.items():

            # Using the features after PCA for Logistic regression alone.
            if(key == "Logistic Regression"):
                x_data = self.x_data_pca
                y_data = self.y_data

            else:
                x_data = self.x_data
                y_data = self.y_data

            model = val_classifier

            # Training the model with cross validation
            accuracy = self.fit_predict_crossvalidate(key,model,x_data,y_data)
            accuracies.append(accuracy)

        return accuracies

    # Training the mode with Cross validation 
    def fit_predict_crossvalidate(self, estimator_name,estimator,x_data,y_data):

        start_time = time.time()

        scoring = ['accuracy', 'f1','precision','recall']

        # 10 fold cross validation(stratified,shuffled)
        scores = cross_validate(estimator, x_data, y_data, cv=10, scoring=scoring)
        avg_accuracy = scores['test_accuracy'].mean()
        avg_f1 = scores['test_f1'].mean()
        avg_precision = scores['test_precision'].mean()
        avg_recall = scores['test_recall'].mean()

        end_time = time.time()

        # calculate total time
        total_time = end_time - start_time

        # display accuracy and time
        print("-------------------------------------")
        print("Cross validation results ")
        print("-------------------------------------")
        print("Classifier:",estimator_name)
        print("Accuracy:", (avg_accuracy * 100), "%")
        print("F1 score:", (avg_f1 * 100), "%")
        print("Precision score:", (avg_precision * 100), "%")
        print("Recall score:", (avg_recall * 100), "%")
        print("Time:", total_time, "sec")
        print("-------------------------------------")

        return avg_accuracy

    
    # Training the models with train test split and without cross validation
    def fit_predict(self):

        x_train, x_test, y_train, y_test , x_train_pca, x_test_pca , y_train_pca, y_test_pca = Load_MushroomDataset_with_Splits()

        classifiers={'Naive Bayes':BernoulliNB(),'Logistic Regression':LogisticRegression(solver='newton-cg')}
        for key,val_classifier in classifiers.items():
            if(key == "Logistic Regression"):
                x_train_data = x_train_pca
                x_test_data = x_test_pca
                y_train_data = y_train_pca
                y_test_data = y_test_pca

            else:
                x_train_data = x_train
                y_train_data = y_train
                x_test_data = x_test
                y_test_data = y_test


            model = val_classifier

            start_time = time.time()
            # train the classifier
            model.fit(x_train_data, y_train_data)
            # test the classifier
            predicted = model.predict(x_test_data)
            # calculating accuracy
            accuracy = accuracy_score(y_test_data, predicted)
            # end time
            end_time = time.time()
            # calculate total time
            total_time = end_time - start_time

             # display accuracy and time
            print("-------------------------------------")
            print("Without Cross validation results ")
            print("-------------------------------------")
            print("Classifier:",key)
            print("Accuracy:", (accuracy * 100), "%")
            print("Time:", total_time, "sec")
            print("-------------------------------------")


def Classify_Mushroom():

    x_data , y_data , x_data_pca = Load_mushroom_dataset()
    mushroom_model = Mushroom(x_data,y_data,x_data_pca)
    mushroom_model.fit_predict()
    accuracy = mushroom_model.classifiers()

    return accuracy

Classify_Mushroom()