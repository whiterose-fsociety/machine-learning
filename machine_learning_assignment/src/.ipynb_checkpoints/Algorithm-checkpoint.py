from dec import train,predict,visualise,acc
import Dataset
from Classifer import *
"""
Algorithm Hyperparameters
"""

"""
Algorithm Tips:
1. If you choose to classify the unknown variables, then "switch off" feature engineering aspects

How It Works

We Have Feature Engineering  & Feature Selection
- . Dealing with Missing Values using three methods
   1. Skip Missing Values
   2. Fill Most Occuring
   3. Apply Classifier
   
   
   Feature Selection
   1. Use chi2 method to use four features
   2. Use mutual information

"""
class Algorithm(Dataset.Visualize):
    def __init__(self, csv):
        super().__init__(csv)
        self.algorithm = "decision_tree"
        self.classifier = Classifier(csv)
        self.fs_features =  ["Tumor-Size","Inv-Nodes","Node-Caps","Irradiat","Class"]
        # self.new_dataset = self.dataset.sample(frac=1)
        self.new_dataset = self.dataset.sample(frac=1)
        self.new_dataset = self.classifier.merge_datasets().sample(frac=1)
        self.new_dataset = self.new_dataset[self.fs_features]
        self.train_dataset,self.test_dataset = self.create_test_train_split()
        self.feature_engineered_train_dataset,self.feature_engineered_test_dataset = self.feature_engineered_test_train_split()
        self.model = train.ID3(self.train_dataset,self.train_dataset)
        self.feature_engineered_model = train.ID3(self.feature_engineered_train_dataset,self.feature_engineered_test_dataset)


    def create_test_train_split(self):
        size_split = len(self.new_dataset) * 75 // 100
        return self.new_dataset[:size_split][features],self.new_dataset[size_split:][features]

    def feature_engineered_test_train_split(self):
        size_split = len(self.new_dataset) * 75 // 100
        features = ["Tumor-Size", "Inv-Nodes", "Node-Caps", "Irradiat", "Class"]
        return self.new_dataset[:size_split][features], self.new_dataset[size_split:][features]

    def model_prediction(self):
        acc.decision_tree(self.test_dataset, self.model)

    def feature_engineered_model_prediction(self):
        acc.decision_tree(self.feature_engineered_test_dataset,self.feature_engineered_model)


# Irradiat, Inv Node, Node-Caps, Deg_Malig,Tumor-Size
# csv = 'breast-cancer.data'

# # features = ["Tumor-Size","Inv-Nodes","Node-Caps","Deg-Malig","Irradiat"]
# features = ["Tumor-Size","Inv-Nodes","Node-Caps","Irradiat","Class"]
# decision_tree = Algorithm(csv)
# decision_tree.feature_engineered_model_prediction()