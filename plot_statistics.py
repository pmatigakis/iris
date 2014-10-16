import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("./data/iris.data", names=["sepal_length", "sepal_width", 
                                          "petal_length", "petal_width", 
                                          "class"])

dataset.boxplot(by="class")

plt.tight_layout()

plt.show()