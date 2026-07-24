import torch
from torch.utils.data import DataLoader
from math import *
from deep_ritz import *
from data_sample import uniform
from plot import plotFunction

if __name__ == "__main__":
    # Hyperparameters

    rep_size = 32
    capacity = 3
    dataset_size = 100000
    batch_size = 64
    interval = [-1, 1]
    it = torch.Tensor(interval)
    epochs = 20
    learning_rate = 1e-3
    bound = torch.Tensor([2, 2])
    decay = 1e-3
    alpha = 1e1

    # Model setup

    dataset = torch.Tensor(uniform(interval[0], interval[1], dataset_size))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork(interval, rep_size=rep_size, capacity=capacity).to(device)
    loss_fn = AreaLoss(bound, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    # Training

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer, it)
    print("Done!")

    torch.save(model, "min_area_model.pth")
    print("Saved PyTorch Model State to min_area_model.pth")
    model.eval()
    
    g = lambda x: 1.679 * cosh(x/1.679)
    print(model(it[0].unsqueeze(dim=0)))
    print(model(it[1].unsqueeze(dim=0)))
    plotFunction(model, g, interval=interval, image=[-5.0, 10])