import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

"""
Round Up Values
"""
def round_up(x):
    return int(math.ceil(x/10.0)) * 10


"""
    Feature Engineering Techniques Used
    
    1. Ignore Missing Values - Might Delete Important Values
    2. Replace With Most Frequent Occuring Values - Creates Imbalanced Dataset
    3. Applying Classifier To Predict Values - 
    4. Apply Unsupervised Learning

"""


"""
TODO: Feature Engineering 
"""
class Dataset:

    # Set The Hyper Parameters
    def __init__(self,csv):
        self.features = ["Class","Age","Menopause","Tumor-Size","Inv-Nodes","Node-Caps","Deg-Malig","Breast","Breast-Quad","Irradiat"]
        self.feature_size = len(self.features)
        self.dataset = self.organize_features(csv)
        self.training_dataset = self.split_data()[0]
        self.testing_dataset = self.split_data()[1]

    def organize_features(self,csv):
        dataset = pd.read_csv(csv,header=None,names=self.features)
        self.features[-1],self.features[0] = self.features[0],self.features[-1]
        dataset = dataset[self.features].replace("?",np.NaN)
        return dataset

    # Perform Feature Engineering
    def clean_data(self):
        pass

    def check_missing(self):
        print(self.dataset.isnull().sum())

    def split_data(self):
        splt_size = len(self.dataset) * 75//100
        training_dataset = self.dataset[:splt_size]
        testing_dataset = self.dataset[splt_size:]
        return [training_dataset,testing_dataset]


    def print_info(self):
        print(self.dataset.head(5))

class Visualize(Dataset):

    """"
    Visualisation Hyperparameters
    """
    def __init__(self,csv):
        super().__init__(csv)
        self.nrows = 3
        self.ncols = 3

    def count_plot(self):
        sns.countplot(data=self.dataset,x='Class',palette='magma')
        plt.show()

    def multi_barplot(self):
        features = self.features[:-1]
        plt.figure(figsize=(16,8))
        for plot in range(1,len(features)+1):
            plt.subplot(3,3,plot)
            self.bar(features[plot-1])
        plt.tight_layout(pad=3.0)
        plt.show()


    def bar(self,feature):
        feature_dict = dict(self.dataset[feature].value_counts())
        X_feature = list(feature_dict.keys())
        Y_feature = list(feature_dict.values())
        Y_pos = np.arange(len(X_feature))
        plt.title("{} Frequency Chart ".format(feature))
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.bar(Y_pos, Y_feature, color=(.5, .1, .5, .6))
        plt.xticks(Y_pos, X_feature)
        max_value = self.dataset[feature].value_counts().max()
        plt.ylim(0, round_up(max_value))


    """
    Do this in pairs
    """
    def crosstab_information(self):
        features = self.features[:-1]
        for feature in features:
            print("Class Target vs {} ".format(feature))
            print(pd.crosstab(self.dataset['Class'],self.dataset[feature],normalize=True))
            print()

    def print_info(self):
        print(self.dataset.head(5))

csv = "breast-cancer.data"
dataset = Dataset(csv)
dataset.check_missing()
visualise = Visualize(csv)
visualise.crosstab_information()
