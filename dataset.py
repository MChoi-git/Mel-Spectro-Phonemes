import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional
import numpy as np
import time



def pad_with(tensor, k, dim):
    """Pads 3d tensor with k zeros pre and the len(longest utterance) + k zeros
    Args:
        tensor (np.array): 3d np object array
        k (int): Number of zeros to pad 
        dim (int): Dimension of tensor (hardcoded for 2 and 3 dims)
    """
    # Get the max length array in tensor (axis=1)
    max_length = 0
    for array in tensor:
        if max_length < array.shape[0]:
            max_length = array.shape[0]
    # Pad every array with 0
    longest_sequence = max_length + k
    new_tensor = []
    for array in tensor:
        if dim == 3:
            npad = (0, 0, k, longest_sequence - len(array))
        if dim == 2:
            npad = (k, longest_sequence - len(array))
        torch_array = torch.tensor(array, device=device)
        padded_array = torch.nn.functional.pad(torch_array, npad)
        new_tensor.append(padded_array)
    new_tensor = torch.nn.utils.rnn.pad_sequence(new_tensor)
    return new_tensor


# Inherit Dataset from torch.utils.data
class MelSpectrogramDataset(torch.utils.data.Dataset):
    """Mel Spectrograms dataset."""
    def __init__(self,
                 dataset_file: str,
                 labels_file: str,
                 hyperp_K: int,
                 transform=None):
        """
        Args:
            dataset_file (string): Path to npy file with mel Spectrograms
            labels_file (string): Path to npy file with labels
            hyperp_K (integer): Hyperparameter value K noting prev/post # of context vectors
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.hyperp_K = hyperp_K
        
        self.spectrograms_array = np.load(dataset_file, allow_pickle=True)  # Load from npy file   
        self.spectrograms_array = pad_with(self.spectrograms_array, self.hyperp_K, 3)  # Pad with zeros
        
        self.label_array = np.load(labels_file, allow_pickle=True)  # Load from npy file
        self.label_array = pad_with(self.label_array, self.hyperp_K, 2)   # Pad with zeros
        
        self.transform = transform

    def __len__(self):
        return len(torch.flatten(self.spectrograms_array, dims=1))


# Initialize cuda
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
x = torch.randn(1).cuda()
torch.cuda.synchronize()

start_time = time.time()
dataset = MelSpectrogramDataset("data/dev.npy", "data/dev_labels.npy", 5, transform=None)
print(dataset.spectrograms_array.shape)
print(dataset.spectrograms_array.device)
print(dataset.label_array.shape)
print("---%s seconds ---" % (time.time() - start_time))
