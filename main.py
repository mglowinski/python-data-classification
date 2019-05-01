import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as nb
import sklearn.model_selection as ms
import sklearn.preprocessing as prep
import sklearn.decomposition as decomp

df = pd.read_csv('iris.data')

# The X variable contains the first four columns of the dataset (attributes) while y contains the labels.
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

# create training and testing variables / Split dataset into training set and test set
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)

# k value for the loop
k_range = range(1, 40)
results = []

# knn for k from k_range
for k in k_range:
    # Instantiate the model with k neighbors.
    knn = nb.KNeighborsClassifier(n_neighbors=k)
    # Fit the model on the training data. Train the model using the training sets
    model = knn.fit(X_train, y_train)
    # Predict the response (output) for a test data.
    predictions = knn.predict(X_test)
    # See how the model performs on the test data.
    results.append(knn.score(X_test, y_test))

print("KNN results: ", results)

#plot results
plt.plot(k_range, results, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=7)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.title('KNN Classification')
plt.show()

################  PCA  ################

# create training and testing vars
X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = ms.train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features / standard scalar normalization to normalize our feature set
scaler = prep.StandardScaler()

# apply transform (map). Compute the mean and standard
X_train_PCA = scaler.fit_transform(X_train_PCA)
X_test_PCA = scaler.transform(X_test_PCA)

# reduction to 2 components
pca = decomp.PCA(n_components=2)

# transform (map) training & testing set
X_train_PCA = pca.fit_transform(X_train_PCA)
X_test_PCA = pca.transform(X_test_PCA)

# variance caused by each of the principal components
explained_variance = pca.explained_variance_ratio_
print('Variance of the principal components', explained_variance)

################  KNN after PCA  ################

resultsPCA = []

# knn for k from k_range
for k in k_range:
    knnPCA = nb.KNeighborsClassifier(n_neighbors=k)
    modelPCA = knnPCA.fit(X_train_PCA, y_train_PCA)
    predictionsPCA = knnPCA.predict(X_test_PCA)
    resultsPCA.append(knnPCA.score(X_test_PCA, y_test_PCA))

print("KNN results after PCA: ", resultsPCA)

# plot results
plt.plot(k_range, resultsPCA, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=7)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.title('KNN Classification after PCA')
plt.show()