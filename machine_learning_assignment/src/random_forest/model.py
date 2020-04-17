from accuracy import *
from dec import *


def proof(conf_matrix):
	ans = input("Are you convinced that it's right:  [y/n]:  ")
	if ans == "n" or ans == "N":
		print()
		show_heatmap(conf_matrix)
	else:
		print("Great !!")


def decision_tree(train_data,test_data):
	# model=  ID3_features(train_data,train_data,features)
	model = ID3(train_data,train_data)
	return model
	
# model = {'Outlook': {'Overcast': 'Yes', 'Rainy': 'No', 'Sunny': {'Temperature': {'Mild': 'No', 'Cool': 'Yes'}}}}
# features = ["Sunny","Hot","High","Weak"]
# ans = predict(model,features)
# print(ans)

# features = ["Sunny","Hot","High","Weak"]
# ans = predict(model,features)
# print
# print(get_leaves(ans))
# print(predict(model,features))
# print(get_leaves(model))

# dataset = pd.read_csv("tennis.csv",header=None,names=["Outlook","Temperature","Humidity","Windy","Play"])
# features=["Outlook","Temperature","Humidity","Windy"]
# split = len(dataset)* 50//100
# train_data = dataset[:split]
# test_data = dataset[split:]
# model = decision_tree(train_data,train_data)
# print(get_leaves(model,features))


