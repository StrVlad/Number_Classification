import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_save_model():
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load data
    train_dataset = torchvision.datasets.MNIST(
        root='./datasets', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./datasets', train=False, download=True, transform=transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    
    # Create and train model
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training (5 epochs as in your code)
    print("Starting training...")
    model.train()
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/5], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/5], Average Loss: {epoch_loss:.4f}')
    
    # Testing
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Model accuracy: {accuracy:.2f}%')
    
    # Save model on CPU for compatibility
    model.to('cpu')
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as 'mnist_model.pth'")
    
    return accuracy

if __name__ == "__main__":
    train_and_save_model()