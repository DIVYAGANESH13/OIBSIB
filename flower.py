from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k_neighbors = int(input("Enter the number of neighbors for the KNN model"))
knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)
print("Enter the measurements of a flower to predict its species.")
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = knn_model.predict(user_input)
species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_species = species_names[prediction[0]]
print(f"\nThe predicted species is: {predicted_species}")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
iris_data = sns.load_dataset("iris")
sns.pairplot(iris_data, hue="species", palette="husl")
plt.suptitle("Pair Plot of Iris Flower Measurements by Species", y=1.02)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X, y)
print("Enter the measurements of a flower to predict its species.")
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))
user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction_proba = knn_model.predict_proba(user_input)[0]
species_names = ['setosa', 'versicolor', 'virginica']
plt.bar(species_names, prediction_proba, color=['blue', 'green', 'purple'])
plt.title("Predicted Probability of Iris Flower Species")
plt.xlabel("Species")
plt.ylabel("Probability")
plt.show()
