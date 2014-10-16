import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = pd.read_csv("./data/iris.data", names=["sepal_length", "sepal_width", 
                                          "petal_length", "petal_width", 
                                          "class"])

pca = PCA(n_components=2)

training_data = dataset[["sepal_length", "sepal_width", 
                         "petal_length", "petal_width"]]

transformed_data = pca.fit_transform(training_data)

iris_setosa_plot = plt.scatter(transformed_data[0:50, 0], transformed_data[0:50, 1], color="r")
iris_versicolour_plot = plt.scatter(transformed_data[50:100, 0], transformed_data[50:100, 1], color="g")
iris_virginica_plot = plt.scatter(transformed_data[100:150, 0], transformed_data[100:150, 1], color="b")

plt.legend([iris_setosa_plot, iris_versicolour_plot, iris_virginica_plot], 
           ["Iris Setosa", "Iris Versicolour", "Iris Virginica"])

plt.show()
