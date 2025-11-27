import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Training data
X_train = np.array([[1,2],[2,3],[3,1],[6,5],[7,7]])
y_train = np.array([0,0,0,1,1])

# Create model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Test data
X_test = np.array([[2,2],[6,6]])

# Predictions
predictions = knn.predict(X_test)
print("Predictions :", predictions)
