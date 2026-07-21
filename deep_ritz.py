import os
import torch
from torch import nn

class Cubic(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.maximum(x**3, torch.zeros_like(x))
    
activations = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "cubic": Cubic
}

class Layer(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        linear = nn.Linear(input_size, output_size)
        if activation != "cubic":
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity=activation)
        else:
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
        self.stack = nn.Sequential(
            linear,
            activations[activation](),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits
    
class ResidualLayer(nn.Module):
    def __init__(self, input_size, in_between, output_size, activation = "cubic"):
        super().__init__()
        self.layer1 = Layer(input_size, in_between, activation)
        self.layer2 = Layer(in_between, output_size, activation)
        self.linear_adjust = nn.Linear(input_size, in_between)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return out + x

class NeuralNetwork(nn.Module):
    def __init__(self, rep_size=32, capacity=3):
        super().__init__()
        activation = "cubic"
        layers = [ResidualLayer(rep_size, rep_size, rep_size, activation) for _ in range(capacity)]
        layers.insert(0, Layer(1, rep_size, activation))
        layers.append(nn.Linear(rep_size, 1))
        self.layer_stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits

class PoisonLoss(nn.Module):
    def __init__(self, aux_func, bound, alpha=1e0):
        super().__init__()
        self.aux_func = aux_func
        self.bound = bound
        self.alpha = alpha
        self.requires_data = True
    
    def forward(self, pred, pred_diff, bound_predict, data):
        y = 0.5 * pred_diff**2 - self.aux_func(data) * pred
        b = bound_predict - self.bound
        return torch.mean(y) + self.alpha * torch.linalg.norm(b)
    
class AreaLoss(nn.Module):
    def __init__(self, bound, alpha=1e0):
        super().__init__()
        self.bound = bound
        self.alpha = alpha
        self.requires_data = False
    
    def forward(self, pred, pred_diff, bound_predict):
        y = pred * torch.sqrt(1 + pred_diff**2)
        b = bound_predict - self.bound
        return torch.mean(y) + self.alpha * torch.linalg.norm(b)


def train(dataloader, model, loss_fn, optimizer, interval):
    model.train()
    total_loss = 0
    for batch, X in enumerate(dataloader):
        X = X.unsqueeze(dim=1)
        X.requires_grad=True
        
        # Compute prediction error
        pred = model(X)
        b_pred = model(torch.unsqueeze(interval, 1)).squeeze(dim=1)
        pred_diff, = torch.autograd.grad(pred, X, grad_outputs=torch.ones_like(X), create_graph=True, retain_graph=True)
        loss = loss_fn(pred, pred_diff, b_pred, X) if loss_fn.requires_data else loss_fn(pred, pred_diff, b_pred)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    avg_loss = total_loss / dataloader.__len__()
    print(f"Average Loss: {avg_loss:>7f}")