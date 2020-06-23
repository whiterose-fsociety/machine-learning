from dec import train,predict,visualise,acc
import Dataset
import pandas as pd
import numpy as np

class Classifier(Dataset.Visualize):
    def __init__(self, csv):
        super().__init__(csv)
        self.p_features = ["Irradiat", "Age", "Menopause", "Tumor-Size", "Inv-Nodes", "Node-Caps", "Deg-Malig", "Breast",
                      "Breast-Quad", "Class"]
        self.missing_value = self.missing_values[1]
        self.x_train, self.x_test, self.c_train, self.c_test = self.split_reorganized_dataset()
        self.re_features = self.reorganize_dataset()

    def reorganize_dataset(self):
        if self.original_features[-1] != 'Class':
            self.original_features[-1], self.original_features[0] = self.original_features[0], self.original_features[-1]
        dataset = self.original_dataset.drop('Class', 1)
        new_features = list(dataset.columns)
        f_index = new_features.index(self.missing_value)
        new_features[f_index], new_features[-1] = new_features[-1], new_features[f_index]
        return new_features

    def input_value_into_test_dataset(self,predicted_values):
        dataset = self.x_test.drop(self.missing_value, 1)
        cols = len(self.x_test.columns)
        dataset.insert(cols - 1, self.missing_value, predicted_values)
        return dataset

    def merge_datasets(self):
        self.x_train = self.x_train[self.re_features]
        self.x_test = self.x_test[self.re_features]
        model = train.ID3(self.x_train, self.x_train)
        predicted_values = predict.predict_dataset(model, self.x_test)
        new_xtest = self.input_value_into_test_dataset(predicted_values)
        new_training_data = pd.concat([self.x_train, self.c_train], axis=1)
        new_testing_data = pd.concat([new_xtest, self.c_test], axis=1)
        new_dataset = pd.concat([new_training_data, new_testing_data], axis=0)
        return new_dataset.sort_index()[self.p_features]


    def split_reorganized_dataset(self):
        train_filter = self.original_dataset[self.missing_value] != "?"
        test_filter = self.original_dataset[self.missing_value] == "?"
        train_dataset = self.original_dataset[train_filter]
        test_dataset = self.original_dataset[test_filter]
        train_class_values = train_dataset.loc[:, 'Class']
        test_class_values = test_dataset.loc[:, 'Class']
        return train_dataset, test_dataset, train_class_values, test_class_values

