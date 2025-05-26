import numpy as np

from decision_tree import DecisionTree
from random_forest_solution import RandomForest
from load_data import generate_data, load_titanic

def main():
    np.random.seed(123)

    train_data, test_data = load_titanic()

    dt = DecisionTree({"depth": 14})
    print("DecisionTree depth 14")
    dt.train(*train_data)
    print("Train")
    dt.evaluate(*train_data)
    print("Test")
    dt.evaluate(*test_data)

    print("random forest")
    rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 14})
    rf.train(*train_data)
    print("Train")
    rf.evaluate(*train_data)
    print("Test")
    rf.evaluate(*test_data)

if __name__=="__main__":
    main()