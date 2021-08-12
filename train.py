from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from model import get_model
from dataset import MelSpectrogramDataset
from torch.utils.data.sampler import SubsetRandomSampler
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_args():
    """Prepare arguments for main"""
    parser = argparse.ArgumentParser(description='Args for training')
    """Optional arguments for pre-split train/test files
    parser.add_argument('--train_file', type=str, default='data/train.npy')
    parser.add_argument('--test_file', type=str, default='data/test.npy')
    parser.add_argument('--label_file',
                        type=str,
                        default='data/train_labels.npy')
    """
    parser.add_argument('--dataset_file', type=str, default='data/dev.npy')
    parser.add_argument('--label_file',
                        type=str,
                        default='data/dev_labels.npy')
    parser.add_argument('--context', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--model_name', type=str, default='SimpleNet')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--shuffle', action='store_true')
    args_to_ret = parser.parse_args()
    return args_to_ret


@torch.no_grad()
def validate(args, model, loss_func, validate_loader):
    """ Validation function"""
    model = model.to(device)
    validation_loader = validate_loader

    val_losses = []

    for val_vector, val_target in validation_loader:
        val_target = val_target.to(dtype=torch.long, device=device)
        pred = model(val_vector.to(dtype=torch.float,
                                   device=device)).transpose(1, 2)
        loss = loss_func(pred, val_target)
        val_losses.append(loss.item())
    print(f"Test loss is {sum(val_losses)/len(val_losses)}")
    return val_losses


def main(args):
    """
    train_dataset = MelSpectrogramDataset(args.train_file, args.label_file, args.context, None, device, True)
    test_dataset = MelSpectrogramDataset(args.test_file, None, args.context, None, device, False)

    """
    dataset = MelSpectrogramDataset(args.dataset_file, args.label_file,
                                    args.context, None, device, None)

    # Split train and test datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=args.shuffle,
                                               num_workers=0,
                                               pin_memory=False)
    validation_loader = torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=args.shuffle,
                                                    num_workers=0,
                                                    pin_memory=False)

    model = get_model(args.model_name, dataset.vector_length).to(device)

    optimizer = getattr(torch.optim, args.optimizer)(model.parameters(),
                                                     lr=args.lr,
                                                     weight_decay=args.wd)

    loss_func = F.cross_entropy

    training_loss = []
    # Train
    for epoch in tqdm(range(args.num_epochs)):

        loss_epoch = []

        for input_vector, label in train_loader:
            label = label.to(dtype=torch.long,
                             device=device,
                             non_blocking=False)

            input_vector = input_vector.to(device, non_blocking=False)
            input_vector = input_vector.float()

            pred = model(input_vector).transpose(1, 2)

            optimizer.zero_grad()

            loss = loss_func(pred, label)

            loss.backward()

            optimizer.step()

            loss_epoch.append(loss.item())

        print(f"Loss at epoch {epoch} is {sum(loss_epoch)/len(loss_epoch)}")
        training_loss.append(sum(loss_epoch) / len(loss_epoch))
    validation_losses = validate(args, model, loss_func, validation_loader)

    # Graph training loss
    y_loss = np.array(training_loss)
    x_epochs = np.arange(1, len(y_loss) + 1)
    sns.set()
    loss_plot = sns.lineplot(x=x_epochs, y=y_loss)
    loss_plot.set(xlabel='Epoch', ylabel='Cross Entropy Loss')
    plt.title('Training Loss')
    plt.show()


if __name__ == "__main__":
    args = prepare_args()
    main(args)
