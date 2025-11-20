import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


X = np.random.rand(500, 10)        # 500 samples, 10 features
y = np.random.randint(0, 2, 500)   # Binary labels (0 or 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1)



class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(10, 16)   # Input: 10 features
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)     # Output: 1 (binary)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = ANN()


criterion = nn.BCELoss()                    # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 100
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")



with torch.no_grad():
    preds = model(X_test)
    preds = (preds > 0.5).float()
    acc = accuracy_score(y_test, preds)
    print("\nAccuracy:", acc)


torch.save(model.state_dict(), r"e_day_12\ann_model.pth")
print("Model saved successfully!")
