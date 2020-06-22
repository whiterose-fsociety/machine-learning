from dec import train,predict,visualise,acc
import dataset
"""
Algorithm Hyperparameters
"""
class Algorithm(dataset.Visualize):
    def __init__(self, csv):
        super().__init__(csv)
        self.algorithm = "decision_tree"
        self.new_dataset = self.dataset.sample(frac=1)
        self.train_dataset,self.test_dataset = self.create_test_train_split()
        self.model = train.ID3(self.train_dataset,self.train_dataset)

    def create_test_train_split(self):
        size_split = len(self.new_dataset) * 75 // 100
        return self.new_dataset[:size_split],self.new_dataset[size_split:]

    def model_prediction(self):
        acc.decision_tree(self.test_dataset, self.model)


# Irradiat, Inv Node, Node-Caps, Deg_Malig,Tumor-Size
csv = 'breast-cancer.data'
decision_tree = Algorithm(csv)
print(decision_tree.model)
decision_tree.model_prediction()
