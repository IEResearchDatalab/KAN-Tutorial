from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import typer
from efficient_kan import KAN
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch_idx: int,
    writer: SummaryWriter,
) -> tuple[float, float]:
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            batch_acc = (output.argmax(dim=1) == labels.to(device)).float().mean()
            epoch_loss += loss.item()
            epoch_acc += batch_acc.item()
            pbar.set_postfix(
                loss=loss.item(),
                accuracy=batch_acc.item(),
                lr=optimizer.param_groups[0]["lr"],
            )
            # TensorBoard batch scalars
            if writer is not None:
                global_step = epoch_idx * len(trainloader) + i
                writer.add_scalar("train/batch_loss", loss.item(), global_step)
                writer.add_scalar("train/batch_acc", batch_acc.item(), global_step)
    n = len(trainloader)
    return epoch_loss / n, epoch_acc / n


def validate(
    model: nn.Module,
    valloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            batch_acc = (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
            val_acc += batch_acc
            all_preds.append(output.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    val_loss /= len(valloader)
    val_acc /= len(valloader)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return val_loss, val_acc, all_preds, all_labels


def plot_metrics(
    out_dir: Path,
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    epoch_idx: int,
):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label="train")
    plt.plot(range(1, len(val_accs) + 1), val_accs, label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"metrics_epoch_{epoch_idx + 1:03d}.png", dpi=150)
    plt.close()


def plot_confusion_matrix(
    out_dir: Path, epoch_idx: int, preds: np.ndarray, labels: np.ndarray
):
    cm = confusion_matrix(labels, preds, labels=list(range(10)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title(f"Confusion matrix epoch {epoch_idx + 1}")
    plt.savefig(out_dir / f"confmat_epoch_{epoch_idx + 1:03d}.png", dpi=150)
    plt.close()


def plot_sample_predictions(
    out_dir: Path,
    model: nn.Module,
    valloader: DataLoader,
    device: torch.device,
    epoch_idx: int,
):
    # take first batch from validation loader
    images, labels = next(iter(valloader))
    n = min(25, images.shape[0])
    imgs = images[:n]
    with torch.no_grad():
        out = model(imgs.view(-1, 28 * 28).to(device))
        preds = out.argmax(dim=1).cpu().numpy()
    # grid using matplotlib
    fig, axes = plt.subplots(5, 5, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(imgs[i].squeeze(), cmap="gray")
            ax.set_title(
                f"T:{int(labels[i])} P:{int(preds[i])}",
                fontsize=8,
                color=("green" if int(labels[i]) == int(preds[i]) else "red"),
            )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"sample_preds_epoch_{epoch_idx + 1:03d}.png", dpi=150)
    plt.close()


def main(num_epochs: int = 10):
    """
    Train a KAN model on MNIST dataset.
    Original code by:
    https://github.com/Blealtan/efficient-kan/blob/master/examples/mnist.py
    You will need to clone that repository and, inside the root folder:
    pip install -e .
    That will install the efficient_kan package needed here.
    """
    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)

    # Define model
    model = KAN([28 * 28, 64, 10])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Define loss
    criterion = nn.CrossEntropyLoss()
    # prepare visualization/logging
    out_dir = Path("output") / "mnist_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # metric storage
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_acc = train_one_epoch(
            model,
            trainloader,
            device,
            optimizer,
            criterion,
            epoch,
            writer,
        )
        val_loss, val_acc, preds, labels = validate(model, valloader, device, criterion)

        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # TensorBoard scalars
        if writer is not None:
            writer.add_scalar("train/loss", epoch_train_loss, epoch)
            writer.add_scalar("train/accuracy", epoch_train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)

        # save plots
        plot_metrics(out_dir, train_losses, val_losses, train_accs, val_accs, epoch)
        plot_sample_predictions(out_dir, model, valloader, device, epoch)
        plot_confusion_matrix(out_dir, epoch, preds, labels)

        # Update learning rate
        scheduler.step()

        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_acc}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    typer.run(main)
