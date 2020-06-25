import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
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
    https://visualstudiomagazine.com/articles/2018/04/01/clustering-non-numeric-data.aspx
   
   Dataset Description
   
   Sources
   http://www.wseas.us/e-library/conferences/2009/istanbul/MAASE/MAASE18.pdf
   Feature Information
   Irradiat -  Exposure To Radiation
   Radiation Therapy Is A Treatment That Uses High-Energy X-Rays To Destroy Cancer cells
   Most of the patients were not exposed to radiation
   
   Age Frequency 
   Most Of the Patients Between Ages 40-69
   
   Premeno ~ Ages [ 40 -69] , It Seems There's A Higher Correlation of Breast Cancer and Age coupled with Premeno(Menopause Transition) 
   Menopause
   premeno - Premenopause or menopause transition, begins several years before menopause.
   Typical age starts for women in their 40s, some start in their 40s
   lt40 - Menopause At The Age Less Than 40 At The Time of Diagnoses
   gt40 - Menopause At The Age Greater Than 40 At The Time of Diagnoses 
   
   A Significant Number Of Patients With Premeno Appeared The Most 
   
   - Extra Information - 
   According To Study https://pubmed.ncbi.nlm.nih.gov/15356404/
   Mean Age of the participants was 50 years
   
    
    Degrees of Malignancy In Cancer of The Breast https://cancerres.aacrjournals.org/content/9/4/453
    Measure The Rate of Growth,Spread and Harm To Cells In The Body At The Time of Diagonses
    The 2nd Degree Appeared Most In The Dataset.
    
    Grade 3 tumors predominately consist of cells that are highly abnormal 


    Inv-Nodes are the number [0-39] of auxilliary lymph nodes that contain metastatic breast cancer visible on historical examination
    metastatic - Pathogenic Agent's Spread From An Initial or Primary Site
    
    Tumor-Size: The Greatest diameter (in mm) of the exercised tumor
    -Most is 30-35
    
    Breast-Quadrant:
    The Breast May Be Divided Into Four Quadrants, Using The Nipple As The Central Point
    - Most is Left_Low Breast
    
    Node-Caps:
    If the cancer does metatstasise to a lymph node, outside of the original site of the tumor.
"""


"""
    Sources
    https://machinelearningmastery.com/feature-selection-with-categorical-data/
    Feature Selection
    High Correlation With Target Variable
    Inv-Nodes 
    The Number of Metastatic Visible Breast Cancer Lymph Nodes Contributes To Classification of Breast Cancer
    Irradiat
    Exposure to Radiation & Previous Radiation Treatments Determine Whether   
    Node-Caps
    Deg-Malig
    
    
    Hypothesis
    Age,Menopause, Breast and Breast-Quad Do Not Have A Significant Contribution To The Class Prediction
    
    
    
    
    Methods For Feature Selection
        • Chi-Squared Feature Selection
        ◦ Hypothesis Test
            ▪ Assumes Null Hypothesis
                • Categorical Observed Frequencies Match Expected Frequencies
            ▪ Degrees of Freedom 
                • (rows -1) * (col - 1)
        ◦ Calculates Statistic That Has Chi-Squared Distribution
        ◦ Test For Categorial Variable Independence
        ◦ Result 
            ▪ Features Independent From Target Are Removed
    • Mutual Information
        ◦ Calculated Between Two Variables 
        ◦ Measures Reduction Uncertainty
        ◦ Measure of Dependence Between Two Variables
        ◦ Correlation Coefficient
    
    
"""
# Python Has This Weird Pointer Thing Going On
class Dataset:

    # Set The Hyper Parameters
    def __init__(self,csv):
        self.features = ["Class","Age","Menopause","Tumor-Size","Inv-Nodes","Node-Caps","Deg-Malig","Breast","Breast-Quad","Irradiat"]
        self.original_features = ["Class","Age","Menopause","Tumor-Size","Inv-Nodes","Node-Caps","Deg-Malig","Breast","Breast-Quad","Irradiat"]
        self.feature_size = len(self.features)
        self.dataset = self.organize_features(csv)
        self.original_dataset = pd.read_csv(csv, header=None, names=self.original_features)
        self.missing_values = ["Node-Caps", "Breast-Quad"]
        # Use Method 1
        self.fill_most()
        self.X,self.Y = self.split_data()
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.Y,test_size=.33,random_state=1)
        self.X_original_train,self.X_original_test,self.Y_original_train,self.Y_original_test = train_test_split(self.X,self.Y,test_size=.33,random_state=1)
        self.X_train_enc = self.x_encode(self.X_train)
        self.X_test_enc  = self.x_encode(self.X_test)
        self.Y_train_enc,self.Y_test_enc = self.y_encode(self.Y_train,self.Y_test)


        # The Number of Features That We Select
        self.k = "all"
        # The Method That Is Used
        self.method = mutual_info_classif
        self.fs = SelectKBest(score_func=self.method,k=self.k)
        self.fs.fit(self.X_train_enc,self.Y_train_enc)
        self.X_train_fs = self.fs.transform(self.X_train_enc)
        self.X_test_fs = self.fs.transform(self.X_test_enc)



##################################################### Dataset Organization #########################################################################################

    # Replace All Question Marks With "?"
    def organize_features(self, csv):
        dataset = pd.read_csv(csv, header=None, names=self.features)
        if self.features[-1] != 'Class':
            self.features[-1], self.features[0] = self.features[0], self.features[-1]
        if True in ["?" in list(dataset[x].unique()) for x in dataset.columns]:
            dataset = dataset[self.features].replace("?", np.NaN)
        return dataset


    def check_missing(self):
        sns.heatmap(self.dataset.isnull(), yticklabels=False, cbar=False, cmap="winter")
        plt.show()

    # After Swapping Columns
    def split_data(self):
        return self.dataset.iloc[:,:-1].astype(str),self.dataset.iloc[:,-1]

    def print_info(self):
        print(self.dataset.head(5))

##################################################### Dataset Organization #########################################################################################

##################################################### Feature Engineering & Missing Values#########################################################################################

    """
     Method 1 of Feature Engineering
     Drop Missing Values
     This Might Delete Important Rows
    """
    def drop_values(self):
        self.dataset.dropna(how="any",inplace=True)

    """
    Method 2 of Feature Engineering
    Populate The Dataset With The Most Occuring Values
    This Creates An Unbalanced Dataset
    Recall, We Have The Hyper Parameters
    """
    def fill_most(self):
        node_caps_top = self.dataset['Node-Caps'].describe()['top']
        breast_quad_top = self.dataset['Breast-Quad'].describe()['top']
        tops = [node_caps_top,breast_quad_top]
        for missing_value,top in zip(self.missing_values,tops):
            if self.dataset[missing_value].isnull().sum() > 0:
                self.dataset[missing_value].replace(np.NaN,top,inplace=True)

    """
    Learn The HyperParameters
    """
    def classifier(self):
        pass

    ##################################################### Feature Engineering & Missing Values#########################################################################################



#####################################################     Feature Selecion #########################################################################################


    """
         Feature Selection: Part 1
         Encode The Feature Data Points
         - Apply Statistical Method To Report Most Appopriate Features 
        """

    def x_encode(self,X):
        for feature in self.X_train.columns:
            l3 = LabelEncoder()
            l3.fit(X[feature])
            X[feature] = l3.transform(X[feature])
        return X

    def y_encode(self,y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc

    #####################################################     Feature Selecion ########################################################################################





class Visualize(Dataset):

    """"
    Visualisation Hyperparameters
    """
    def __init__(self,csv):
        super().__init__(csv)
        self.nrows = 3
        self.ncols = 3

    def count_plot(self):
        sns.countplot(x='Class',data=self.dataset)
        plt.show()

    def multi_barplot(self):
        features = self.features[:-1]
        plt.figure(figsize=(16,8))
        for plot in range(1,len(features)+1):
            plt.subplot(3,3,plot)
            self.bar(features[plot-1])
        plt.tight_layout(pad=3.0)
        plt.show()

    def feature_contribution(self):
        plt.figure(figsize=(20,20))
        for i,feature in zip(range(len(self.fs.scores_)),self.features):
            print("{} : {}".format(feature,self.fs.scores_[i]))
        plt.bar([i for i in range(len(self.fs.scores_))], self.fs.scores_)
        Y_pos = np.arange(len(self.features))
        plt.xticks(Y_pos, self.features[:-1])
        plt.title("Contribution To Cancer Determination")
        plt.xlabel("Features")
        plt.ylabel("Contribution Frequency")
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
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']},index=[0, 1, 2, 3])
# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],'B': ['B4', 'B5', 'B6', 'B7'],'C': ['C4', 'C5', 'C6', 'C7'],'D': ['D4', 'D5', 'D6', 'D7']},index=[4, 5, 6, 7])
# df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],'B': ['B8', 'B9', 'B10', 'B11'],'C': ['C8', 'C9', 'C10', 'C11'],'D': ['D8', 'D9', 'D10', 'D11']},index=[8, 9, 10, 11])
# frames = [df1,df2,df3]
