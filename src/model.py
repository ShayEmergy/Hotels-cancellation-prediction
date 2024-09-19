import torch
import torch.nn as nn

class HotelCancellationModel(nn.Module):
    def __init__(self, input_size):
        super(HotelCancellationModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 74),
            nn.ReLU(),
            nn.Linear(74, 36),
            nn.ReLU(),
            nn.Linear(36, 3)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, X_train, y_train, num_epochs=500, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean()
    return accuracy.item()
