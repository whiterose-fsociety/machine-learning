import model
import pandas as pd
import dec
import accuracy
from statistics import mode
# Fetch data
dataset = pd.read_csv("tennis.csv",header=None,names=["Outlook","Temperature","Humidity","Windy","Play"])
features=["Outlook","Temperature","Humidity","Windy"]
# Makes it fair
dataset = dataset.sample(frac=1)

def split_data(dataset):
	split = len(dataset) * 50//100
	train = dataset.iloc[:split].reset_index(drop=True)
	return train


def random_forest_model(dataset,num_trees):
	random_forest = []
	for i in range(num_trees):
		bootstrap_sample = dataset.sample(frac=1,replace=True)
		bootstrap_train = split_data(bootstrap_sample)
		tree = model.decision_tree(bootstrap_train,bootstrap_train)
		random_forest.append(tree)
	return random_forest


# Returns the decision given the query
def forest_predict(forest,query):
	predicts = []
	for i in range(len(forest)):
		# if prediction is a dictionary that means, the tree does not give a value 
		prediction = accuracy.predict(forest[i],query)
		filt = filter_pred(prediction)
		predicts.append(filt)
	return predicts


# Prints each decision tree and it's decision based on the query
def print_forest(forest,query):
	for i in range(len(forest)):
		print(i ,": ",forest[i])
		prediction = model.predict(forest[i],query)
		filt = filter_pred(prediction)
		print(i , ": Prediction: ",filt)
		print()


# Filter the prediction to return a "Yes" or "No"
def filter_pred(prediction):
	if type(prediction) == list:
		if type(prediction[0]) == dict:
			pred = dec.get_leaves(prediction[0])
			return dec.get_mode(pred)
		else:
			return prediction[0]
	elif type(prediction) == dict:
		pred = dec.get_leaves(prediction)
		return dec.get_mode(pred)
	else:
		return prediction

def predict_testing_data(forest,test_data):
	prediction_class = []
	for row in range(len(test_data)):
		query = list(test_data.iloc[row,:-1])
		decision = forest_predict(forest,query)
		prediction_class.append(dec.get_mode(decision))
	return prediction_class


def accurate_testing():
	pass

split = len(dataset) * 50//100
testing_data = dataset.iloc[split:].reset_index(drop=True)
print(testing_data)
forest = random_forest_model(dataset,10)
query = list(testing_data.iloc[0,:-1])
q = ["Rainy","Mild","Normal","Weak","Yes"]
actual_data = list(testing_data.iloc[:,-1])
predicted_data = predict_testing_data(forest,testing_data)
print("Actual Data",actual_data)
print("Predicted Data",predicted_data)

# Generalise later
actual_n = accuracy.convert_first(actual_data)
predicted_n = accuracy.convert_first(predicted_data)
print("Actual",actual_n)
print("Predicted",predicted_n)


data = {"y_act":actual_n,"y_pred":predicted_n}
df = pd.DataFrame(data,columns = ["y_act","y_pred"])
conf_matrix = pd.crosstab(df["y_act"],df["y_pred"],rownames=["Actual"],colnames=["Predicted"],margins=True,margins_name="Total")
print()
print(conf_matrix)
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
FP = conf_matrix[1][0]
FN = conf_matrix[0][1]
acc = accuracy.accuracy(TP,TN,FP,FN)
recall = accuracy.recall(TP,FN)
precision = accuracy.precision(TP,FP)
f_measure_data = accuracy.f_measure(recall,precision)
print("Accuracy",acc)
print("Recall",recall)
print("Precision",precision)
print("F Measure",f_measure_data)
accuracy.show_heatmap(conf_matrix)
# recall_data = accuracy.recall(TP,FN)
# precision_data = accuracy.precision(TP,FP)
# print(acc)
# print(recall_data)
# print()
# print([{"Accuracy":acc, "Recall":recall_data,"Precision":precision_data,"F Measure":f_measure_data},conf_matrix])
