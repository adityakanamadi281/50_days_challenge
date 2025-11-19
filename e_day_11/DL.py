import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np



x = np.random.rand(100, 64, 32, 3)       
y = np.random.randint(0, 2, size=(100, 1)) 


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)



x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


x_train = x_train.view(x_train.size(0), -1)
x_test = x_test.view(x_test.size(0), -1)

input_dim = x_train.shape[1]


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = ANN()


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 150
for epoch in  range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


with torch.no_grad():
    y_pred = model(x_test)
    y_pred = (y_pred > 0.5).float()


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)