import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

dataset = pd.read_csv("./data/iris.data", names=["sepal_length", "sepal_width", 
                                          "petal_length", "petal_width", 
                                          "class"])

training_data = dataset[["sepal_length", "sepal_width", 
                         "petal_length", "petal_width"]]

target = dataset["class"]

classes = set(target.values)

target = target.replace(classes, range(len(classes)))

clf = LogisticRegression()

scores = cross_val_score(clf, training_data, target, cv=10)

score_statistics = pd.Series(scores).describe()

print score_statistics