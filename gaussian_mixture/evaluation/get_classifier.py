import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 28 -> 14

            nn.Conv2d(32, 64, 3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 14 -> 7

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)
    

def get_mnist_classifier(
    batch_size=128,
    lr=1e-3,
    epochs=5,
    device=None,
    verbose=True
):
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root="./mnist", train=True, download=True, transform=transform)
    test_data  = datasets.MNIST(root="./mnist", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --------------------
    #   Training
    # --------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

    # --------------------
    #   Evaluation
    # --------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total

    if verbose:
        print(f"Test accuracy: {accuracy*100:.2f}%")

    return model, accuracy


def main():
    # -------------------------
    #  Create checkpoints dir
    # -------------------------
    root_ckpt = "./checkpoints/mnist_classifier/"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_dir = os.path.join(root_ckpt, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # -------------------------
    #  Train classifier
    # -------------------------
    print("Training MNIST classifier...")
    model, accuracy = get_mnist_classifier(
        batch_size=128,
        lr=1e-3,
        epochs=5,
        device=None,
        verbose=True
    )

    # -------------------------
    #  Save model weights
    # -------------------------
    ckpt_path = os.path.join(save_dir, "classifier.pth")
    torch.save(model.state_dict(), ckpt_path)

    # -------------------------
    #  Save accuracy
    # -------------------------
    with open(os.path.join(save_dir, "accuracy.txt"), "w") as f:
        f.write(f"{accuracy:.6f}")

    print("\nDone.")
    print(f"Model saved to:      {ckpt_path}")
    print(f"Accuracy saved to:   {save_dir}/accuracy.txt")
    print(f"Test Accuracy:       {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
