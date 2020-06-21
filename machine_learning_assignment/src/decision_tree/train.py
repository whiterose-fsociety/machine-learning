import pandas as pd 
import numpy as np


 ################################################################################################################################3
"""
 TRAINING THE MODEL
"""
########################################### Correct Methods ###########################################
def entropy(dataset):
  unique_values =  list(dataset.iloc[:,-1].value_counts())
  probs = [x/sum(unique_values) for x in unique_values]
  entropy = sum([prob*np.log2(prob) for prob in probs])
  return abs(entropy)


def entropy_feature(dataset,column):
  unique_values = list(dataset[column].value_counts())
  probs = [x/sum(unique_values) for x in unique_values]
  
  entropy = sum([prob*np.log2(prob) for prob in probs])
  return abs(entropy)


def info_gains(dataset):
	cols = dataset.shape[1]
	info_gains = {}
	for col in range(cols-1):
		unique_values = list(dataset.iloc[:,col].value_counts())
		unique_index = dataset.iloc[:,col].value_counts().index.tolist()

		info_gain = 0
		data_entropy = entropy(dataset)
		data_size = dataset.shape[0]

		entropies = []

		for unique_feature in unique_index:
			filt = dataset.iloc[:,col] == unique_feature
			feature_dataset = dataset[filt]
			entropies.append((entropy(feature_dataset),feature_dataset.shape[0]))

		branch_entropy = sum([ data[0] * data[1] for data in entropies])
		info_gain = data_entropy - branch_entropy/data_size		

	
		info_gains[col] = info_gain
	return info_gains


def best_attribute(gains):
	x  = []
	for gain in gains.values():
		x.append(gain)
	m = max(x)
	index = list(gains.keys())[list(gains.values()).index(m)]
	return index


def info_gain_feature(dataset,feature):
	
	unique_values = list(dataset[feature].value_counts())
	
	unique_index = dataset[feature].value_counts().index.tolist()




	info_gain = 0
	data_entropy = entropy(dataset)
	data_size = dataset.shape[0]

	entropies = []

	for unique_feature in unique_index:
		filt = dataset[feature]	== unique_feature
		feature_dataset = dataset[filt]	
		entropies.append((entropy(feature_dataset),feature_dataset.shape[0]))


	branch_entropy = sum([ data[0] * data[1] for data in entropies])
	info_gain = data_entropy - branch_entropy/data_size
	
	return info_gain

def info_gains_features(dataset,features):
	info_gains = {}
	feature_size = len(features)
	for feature,size in zip(features,range(feature_size)):
		info_gains[feature] = info_gain_feature(dataset,feature)
	return info_gains
 
def ID3_features(sub_dataset,dataset,features,parent_node_target=None):
	#If target values have the same value
	if len(np.unique(sub_dataset.iloc[:,-1])) <=1 :
		return np.unique(sub_dataset.iloc[:,-1])[0]
	#If sub_dataset is empty	
	elif len(sub_dataset) == 0:
		return parent_node_target

	#If feature list is empty
	elif len(features) == 0:
		target_dict = sub_dataset.iloc[:,-1].value_counts().to_dict()
		return best_attribute(target_dict)
	else:
		parent_node_dict = sub_dataset.iloc[:,-1].value_counts().to_dict()
		parent_node_target = best_attribute(parent_node_dict)
		gains = info_gains_features(sub_dataset,features)
		best_feature = best_attribute(gains)

		tree = {best_feature:{}}


		features.remove(best_feature)

		# Grow a branch under the root
		feature_values = sub_dataset[best_feature].value_counts().index.tolist()
		for feature_value in feature_values:
			filt = sub_dataset[best_feature] == feature_value
			sub = sub_dataset[filt]
			subtree = ID3_features(sub,dataset,features,parent_node_target)

			tree[best_feature][feature_value] = subtree

		return (tree)

def ID3(sub_dataset,dataset):
	features = list(dataset.columns)
	features.remove(features[-1])
	return ID3_features(sub_dataset,dataset,features)
########################################### Correct Methods ###########################################


################################################################################################################################3

