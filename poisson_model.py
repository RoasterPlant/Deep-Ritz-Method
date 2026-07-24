import torch
from torch.utils.data import DataLoader
from math import *
from deep_ritz import *
from data_sample import uniform
from plot import plotFunction

if __name__ == "__main__":
    # Hyperparameters

    rep_size = 64
    capacity = 5
    dataset_size = 100000
    batch_size = 64
    interval = [0, 1]
    it = torch.Tensor(interval)
    epochs = 20
    learning_rate = 1e-3
    #f = lambda x: torch.sin(x)
    f = lambda x: torch.exp(x)
    bound = torch.Tensor([1, 2])
    decay = 1e-3
    alpha = 1e1

    # Model setup
    
    dataset = torch.Tensor(uniform(interval[0], interval[1], dataset_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork(interval, rep_size, capacity).to(device)
    loss_fn = PoisonLoss(f, bound, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    # Training

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer, it)
    print("Done!")

    torch.save(model.state_dict(), "poisson_model.pth")
    print("Saved PyTorch Model State to poisson_model.pth")
    model.eval()

    
    g = lambda x: - exp(x) + e * x + 2
    #g = lambda x: x + 1 
    #g = lambda x: sin(x)
    print(model(it[0].unsqueeze(dim=0)))
    print(model(it[1].unsqueeze(dim=0)))
    plotFunction(model, g, interval=interval, image=[0.5, 3])