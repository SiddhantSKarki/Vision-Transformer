import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.nn import functional as F
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import argparse
import tqdm

import pickle
from torch.utils.tensorboard import SummaryWriter
import os
import re
import glob

from VisionTransformer import vit, config
from torchvision.datasets import CIFAR10



device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad
def evaluate(model, test_loader, eval_func, avg=None):
    model.eval()
    accuracy = 0.0
    for batch in test_loader:
        tensors, labels = batch
        tensors, labels = tensors.to(device), labels.to(device)
        logits = model(tensors)
        predictions = torch.argmax(logits, axis=-1).to('cpu')
        labels = labels.to('cpu')
        if avg is None:
            accuracy += eval_func(labels, predictions)
        else:
            accuracy += eval_func(labels, predictions, average=avg, zero_division=0.0)
    model.train()
    return accuracy / len(test_loader)

def clear_screen():
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/MacOS
        os.system('clear')

def search_checkpoint(dir):
    epochs = glob.glob(os.path.join(dir,"*.pt"))
    if len(epochs) == 0:
        return None
    epochs = [os.path.basename(e) for e in epochs]
    epochs = [int(re.match(r"([\d]*)(?=.pt)", names).group(1)) for names in epochs]
    return max(epochs)

def train(configs,
        train_loader, test_loader, epochs, eval_iter, log_dir, checkpoint_dir, lr=1e-4):
    
    saved_epoch = search_checkpoint(checkpoint_dir)

    model = vit.VisionTransformer(configs)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    iteration = 0
    if saved_epoch is not None:
        print(f"Checkpoint Found. Loading model from epoch {saved_epoch}")
        model_path = os.path.join(checkpoint_dir, f"{saved_epoch}.pt")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        saved_epoch = 0
        
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

    clear_screen()
    for epoch in range(saved_epoch, epochs + 1):
        running_loss = 0.0
        with tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{(epochs + 1)}", leave=True) as pbar:
            for batch in train_loader:
                tensors, labels = batch
                tensors, labels = tensors.to(device), labels.to(device)
                logits = model(tensors)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # print(f"Epoch {epoch}, curr loss: {loss.item()}")
                loss = loss.item()
                writer.add_scalar("Loss/train_batch", loss, iteration)
                running_loss += loss
                pbar.update(1)
                iteration += 1
        acc = evaluate(model, test_loader, accuracy_score).round(2)
        # pre = evaluate(model, test_loader, precision_score, avg='weighted').round(2)
        # rec = evaluate(model, test_loader, recall_score, avg='weighted').round(2)
        writer.add_scalar("val?acc", acc, epoch)
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            'step': iteration
        }, os.path.join(checkpoint_dir, f"{epoch}.pt"))
        # if epoch % eval_iter == 0 and epoch > 0:
        print(f"Epoch {epoch}, \
                curr loss: {running_loss}, \
                mean_accuracy: {acc}, \
                mean_precision: {pre}, \
                mean_recall: {rec}")



if __name__ == '__main__':
    #TODO: Convert this to CLA and add a shell file for training
    #TODO: Add a JSON 
    input_channels = 3 # 1 for Gray Scale Imgs and 3 for RGB
    num_classes = 100
    p_embd_size = 4
    batch_size = 5
    img_size = 256
    kernel_size = 16
    n_patches = (img_size // kernel_size)**2
    fp = torch.float32
    num_heads = 4
    num_blocks = 4

    lr = 1e-4
    eval_iter = 5
    epochs = 1000

    checkpoint_dir = "../checkpoints"
    log_dir = "../logs"

    if not os.path.exists(checkpoint_dir):
        print("Creating Checkpoint Directory...")
        os.mkdir(checkpoint_dir)
    if not os.path.exists(log_dir):
        print("Creating Logs Directory...")
        os.mkdir(checkpoint_dir)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),  # Convert to 3 channels if not already
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    training_dataset = CIFAR10(root="../cifar10/", download=True, transform=transform)
    
    test_dataset = CIFAR10(root="../cifar10/", download=True, train=False, transform=transform)

    train_loader = data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    t_config = config.ViTConfig(
        input_channels=input_channels,
        num_classes=num_classes,
        embedding_size=p_embd_size,
        patch_size=kernel_size,
        num_patches=n_patches,
        num_heads=num_heads,
        num_blocks=num_blocks,
        precision=fp,
        batch_size=batch_size,
        device=device
    )

    train(configs=t_config,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        eval_iter=eval_iter,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir)

    
