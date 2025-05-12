import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import shutil

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_and_save_model():
    # Define transformations for the training data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model, loss function and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model for 5 epochs
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    # Save the model to the finalproject-streamlit directory
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'finalproject-streamlit')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'mnist_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model trained and saved as {save_path}")

    # Additionally, copy the model file to the current working directory (where Streamlit app runs)
    current_dir = os.getcwd()
    dest_path = os.path.join(current_dir, 'mnist_model.pth')
    shutil.copyfile(save_path, dest_path)
    print(f"Model file copied to {dest_path}")

if __name__ == "__main__":
    train_and_save_model()
