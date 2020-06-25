
# BEHOLD THE POWER OF ABSTRACTION

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
from statistics import mode
from nested_lookup import nested_lookup
import dec




# Predicts the output given the features
# Given the conditions: Can we go play tennis
# predict(tree = {"Outlook":{...} ,        features = ["Sunny","Hot","High","Weak"] )
# returns "No"
def predict(tree,features):
	for feature in features:
		new_tree = nested_lookup(feature,tree)
		if len(new_tree) == 0:
			continue
		else:
			tree = new_tree
	return tree
 # given a list of "Yes" and "No" , return a list of 1's and 0'ss
 # Generalised case
def convert(class_list):
	class_unique = np.unique(class_list)
	class_dict = {}
	for index in range(len(class_unique)):
		class_dict[class_unique[index]] = index

	class_list =  [class_dict[item] for item in class_list]
	return class_list



# works 
def convert_first(class_list):
	class_dict = {"Yes":1,"No":0}
	class_list = [class_dict[item] for item in class_list]
	return class_list


#Accuracy
#Assumes cost
def accuracy(TP,TN,FP,FN):
	numerator = TP + TN
	denominator = TP + TN + FP + FN
	return numerator/denominator
	

# When it actually says, i.e real data
#how often does it predict yes
def recall(TP,FN):
  return TP/(TP+FN)


#When it predicts yes , how often
def precision(TP,FP):
	return TP/(TP+FP)




def f_measure(recall,precision):
	numerator = 2* recall * precision
	denominator = recall + precision
	return numerator / denominator


def predict_dataset(model,test_data):
	class_list = []
	for i in range(len(test_data)):
		sample = list(test_data.iloc[i,:-1])
		class_list.append(predict(model,sample)[0])
	return class_list

# l = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# print("General algorithm" , convert(l))
def prediction(actual,predicted):
	actual_n = convert(actual)
	predicted_n = convert(predicted)

	data = {"y_act":actual_n,"y_pred":predicted_n}
	df = pd.DataFrame(data,columns = ["y_act","y_pred"])
	conf_matrix = pd.crosstab(df["y_act"],df["y_pred"],rownames=["Actual"],colnames=["Predicted"],margins=True,margins_name="Total")
	
	TP = conf_matrix[1][1]
	TN = conf_matrix[0][0]
	FP = conf_matrix[1][0]
	FN = conf_matrix[0][1]
	acc = accuracy(TP,TN,FP,FN)
	recall_data = recall(TP,FN)
	precision_data = precision(TP,FP)
	f_measure_data = f_measure(recall_data,precision_data)
	return [{"Accuracy":acc, "Recall":recall_data,"Precision":precision_data,"F Measure":f_measure_data},conf_matrix]


def dict_print(dic):
	for k,v in dic.items():
		print(k," : " ,v)

def show_heatmap(conf_matrix):
	sn.heatmap(conf_matrix,annot=True)
	plt.title("Confusion Matrix: Are you convinced yet ?")
	plt.show()


