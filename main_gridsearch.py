import torch
import numpy as np
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from torchvision import datasets, transforms
from src.cnn_pytorch import FashionCNN

def run_grid_search():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)

    #use a subset for the grid search to save time
    X = train_data.data.unsqueeze(1).float() / 255.0
    y = train_data.targets

    net = NeuralNetClassifier(
        module=FashionCNN,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        max_epochs=10,
        batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0 # Keep it quiet during grid search
    )

    #define the Grid
    param_grid = {
        'lr': [0.001, 0.01],
        'module__dropout': [0.3, 0.5],
        'module__hidden_units': [128, 256],
        'max_epochs': [10, 20]
    }

    # 4. Run Grid Search
    gs = GridSearchCV(net, param_grid, refit=True, cv=3, scoring='accuracy', verbose=2)

    print("--- Starting GridSearchCV ---")
    gs.fit(X, y)

    print(f"\nBest Parameters: {gs.best_params_}")
    print(f"Best Score: {gs.best_score_:.4f}")

if __name__ == "__main__":
    run_grid_search()
