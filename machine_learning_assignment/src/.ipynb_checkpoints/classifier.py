from dec import train,predict,visualise,acc
import pandas as pd
import numpy as np


# change_features = ['Node-Caps','Breast-Quad']
def reorganize_dataset(dataset,missing_feature,features):
    if features[-1] != 'Class':
        features[-1], features[0] = features[0],features[-1]
    dataset = dataset.drop('Class',1)
    new_features = list(dataset.columns)
    f_index = new_features.index(missing_feature)
    new_features[f_index],new_features[-1] = new_features[-1],new_features[f_index]
    return new_features


def split_reorganized_dataset(dataset,missing_feature):
    train_filter = dataset[missing_feature] != "?"
    test_filter = dataset[missing_feature] == "?"
    train_dataset = dataset[train_filter]
    test_dataset = dataset[test_filter]
    train_class_values = train_dataset.loc[:,'Class']
    test_class_values = test_dataset.loc[:,'Class']
    return train_dataset,test_dataset,train_class_values,test_class_values

def input_value_into_test_dataset(test_dataset,missing_feature,predicted_values):
    dataset = test_dataset.drop(missing_feature,1)
    cols = len(test_dataset.columns)
    dataset.insert(cols-1,missing_feature,predicted_values)
    return dataset



def merge_datasets(xtrain,xtest,c_train,c_test,missing_value):
    xtrain = xtrain[re_features]
    xtest = xtest[re_features]
    model = train.ID3(xtrain, xtrain)
    predicted_values = predict.predict_dataset(model, xtest)
    new_xtest = input_value_into_test_dataset(xtest, missing_value, predicted_values)
    new_training_data = pd.concat([xtrain, c_train], axis=1)
    new_testing_data = pd.concat([new_xtest, c_test], axis=1)
    new_dataset = pd.concat([new_training_data, new_testing_data], axis=0)
    return new_dataset.sort_index()


features = ["Class","Age","Menopause","Tumor-Size","Inv-Nodes","Node-Caps","Deg-Malig","Breast","Breast-Quad","Irradiat"]
p_features = ["Irradiat","Age","Menopause","Tumor-Size","Inv-Nodes","Node-Caps","Deg-Malig","Breast","Breast-Quad","Class"]
csv = 'breast-cancer.data'
dataset = pd.read_csv(csv,header=None,names=features)
missing_value = "Node-Caps"
xtrain,xtest,c_train,c_test = split_reorganized_dataset(dataset,missing_value)
# Reorganize Features
re_features = reorganize_dataset(dataset,missing_value,features)
classified_dataset = merge_datasets(xtrain,xtest,c_train,c_test,missing_value)
classified_dataset = classified_dataset[p_features]
print("Classified")
print(classified_dataset)
print("Original")
print()
print(dataset[p_features])
