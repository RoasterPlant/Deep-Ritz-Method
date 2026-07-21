import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from math import *
from deep_ritz import *

def plotFunction(f, g = None, xLabel = "x", yLabel = "y", interval = [-1, 1], image = None, title = "", MAX_ITER = 1000):
    x = np.linspace(interval[0], interval[1], MAX_ITER)
    xt = torch.Tensor(x)
    y = np.array([f(i.unsqueeze(dim=0)).item() for i in xt])
    fig, ax = plt.subplots()
    line1, = ax.plot(x, y)
    line1.set_label('Solução Aproximada')
    if g != None:
        y2 = np.array([g(i) for i in x])
        line2, = ax.plot(x, y2)
        line2.set_label('Solução Analítica')
    ax.set_xlim(interval[0], interval[1])
    if image != None:
        ax.set_ylim(image[0], image[1])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.show()

def plotParametricCurve(fx, fy, xLabel = "x", yLabel = "y", interval = [0, 1], image = None, title = "", MAX_ITER = 1000):
    t = np.linspace(interval[0], interval[1], MAX_ITER)
    x = np.array([fx(i) for i in t])
    y = np.array([fy(i) for i in t])

    fig, ax = plt.subplots()
    line1, = ax.plot(x, y)
    ax.set_xlim(image[0], image[1])
    if image != None:
        ax.set_ylim(image[0], image[1])
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    #fx = lambda t: cos(t)
    #fy = lambda t: sin(t)
    #plotParametricCurve(fx, fy, interval=[0, 2 * pi], image=[-1.5, 1.5])
    #f = lambda z: 1.697 * cosh(z/1.697)
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    print(model(torch.zeros(1)))
    print(model(torch.ones(1)))
    plotFunction(model, interval=[0, 1], image=[0.5, 3])