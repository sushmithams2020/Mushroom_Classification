import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2,SelectKBest

# Dataset for cross validation
def Load_mushroom_dataset():

    #loading mushoroom dataset from UCI
    path="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    df = pd.read_csv(path)

    column = ['class', 'cap shape', 'cap surface', 'cap color', 'bruised', 'odor','gill attachment', 'gill spacing',
               'gill size', 'gill color', 'stalk shape', 'stalk root', 'stalk surface above ring','stalk surface below ring',
               'stalk color above ring','stalk color below ring', 'veil type', 'veil color', 'ring number','ring type',
               'spore print color', 'population', 'habitat']
    df.columns = column

    x_data = df.copy(deep=True)
    x_data.drop(['class'], inplace=True, axis=1)

    x_data.drop(['veil type'], inplace=True, axis=1)   #column veil-type has the same value for all samples.So this is removed.
    y_data = df['class'].copy(deep=True)

    # Converting Categorical features into binary values
    x_data = pd.get_dummies(x_data)

    # Converting the categorical labels into integers
    label_Encoder = LabelEncoder()
    y_data = label_Encoder.fit_transform(y_data)        #label 'e' and 'p' are assigned as 0 and 1 respectively.

    # PCA considering 98% variance
    x_data_pca = x_data
    pca = PCA(n_components = 0.98)
    sc = StandardScaler()
    x_data_pca = sc.fit_transform(x_data_pca)
    x_data_pca = pca.fit_transform(x_data_pca)    # Reduced to 68 features.

    return x_data , y_data , x_data_pca

# Dataset split to check for the accuracies without cross validation
def Load_MushroomDataset_with_Splits():
    x_data , y_data , x_data_pca = Load_mushroom_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=42)
    x_train_pca, x_test_pca , y_train_pca, y_test_pca= train_test_split(x_data_pca, y_data, test_size=0.20, random_state=42)

    return x_train, x_test, y_train, y_test , x_train_pca, x_test_pca , y_train_pca, y_test_pca

# Dataset split for Random Forest classifier with train test split
def Load_mushroom_dataset_RF():
    x_data , y_data , x_data_pca = Load_mushroom_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=42)

    return x_train, x_test, y_train, y_test

# Exploring the dataset and cleaning the dataset.
def display_dataset_info():
    path="https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    df = pd.read_csv(path)

    column = ['class', 'cap shape', 'cap surface', 'cap color', 'bruised', 'odor','gill attachment', 'gill spacing',
               'gill size', 'gill color', 'stalk shape', 'stalk root', 'stalk surface above ring','stalk surface below ring',
               'stalk color above ring','stalk color below ring', 'veil type', 'veil color', 'ring number','ring type',
               'spore print color', 'population', 'habitat']
    df.columns = column

    print(df.head())

    # Categories in each feature 
    column_list = df.columns.values.tolist()
    for column_name in column_list:
        print(column_name)
        print(df[column_name].unique())

    # Categories in labels
    print('classes')
    print(df['class'].unique())

    # To find the feature that doesnt influence much in classification
    print(df.apply(pd.Series.nunique))   # column veil-type has the same value for all samples.So this can be removed.
    
display_dataset_info()
