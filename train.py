import torch
import torch.nn as nn
from model import Network


def train_one_epoch(model, optimizer, criterion, train_loader, val_loader, save_path):
    pass


def train(model_conf_path, optimizer_type, lr, criterion_type, num_epochs, train_loader, val_loader=None, device=None,
          save_path=None):
    # init model
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Network(model_conf_path).to(device)
    model.train()
    criterion = getattr(nn, criterion_type)()
    optimizer = getattr(nn, optimizer_type)(lr=lr)
    for epoch in range(num_epochs):
        train_loader(model, optimizer, criterion, train_loader, val_loader, save_path)
    return model
