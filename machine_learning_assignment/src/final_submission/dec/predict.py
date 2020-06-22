import random


def print_test(drop_filter_dataset,model):
	print("Dataset")
	print(drop_filter_dataset)
	print()
	print("Model")
	print(model)
	print()

####################################################### Correct Methods #######################################################


# query = {'Irradiat': 'yes', 'Age': '50-59', 'Menopause': 'ge40', 'Tumor-Size': '30-34', 'Inv-Node': '6-8', 'Node-Caps': 'yes', 'Deg-Malig': 2, 'Breast': 'left', 'Breast-Quad': 'right_low'}
# model = {'Node-Caps': {'no': {'Tumor-Size': {'25-29': {'Irradiat': {'no': 'no-recurrence-events', 'yes': 'recurrence-events'}}, '15-19': 'no-recurrence-events', '20-24': 'no-recurrence-events', '30-34': {'Age': {'60-69': 'no-recurrence-events', '40-49': 'recurrence-events'}}, '10-14': 'no-recurrence-events', '35-39': 'recurrence-events', '50-54': 'no-recurrence-events'}}, 'yes': 'recurrence-events'}}
# Might have to do some crazy ninja pruning
def predict(query,model,default = "no-recurrence-events"):
    events = ["no-recurrence-events","recurrence-events"]
    for key in list(query.keys()):
    	#Check if the feature name exists in the model features
        if key in list(model.keys()):
            try:
            	# Attempt to get the value of the "first" feature
                result = model[key][query[key]]
            except:
            	#If the model does have the value, return 1
            	#Return down the stack trace
                return random.choice(events)
            #Get the resulting model or value
            #If it's model then keep going
            #else just return the result
            result = model[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result

def predict_dataset(model,test_data):
	prediction_values = []
	queries = test_data.iloc[:,:-1].to_dict(orient="records")
	for i in range(len(test_data)):
		prediction_values.append(predict(queries[i],model))	
	return prediction_values



####################################################### Correct Methods #######################################################







