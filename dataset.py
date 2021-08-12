import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.nn import functional
import numpy as np
import time


def main():
    torch.cuda.empty_cache
    # Initialize cuda
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    x = torch.randn(1).cuda()
    torch.cuda.synchronize()

    start_time = time.time()
    dataset = MelSpectrogramDataset("data/dev.npy", "data/dev_labels.npy", 1,
                                    None, device)

    # Running quick tests, to be replaced by pytest later
    # Check shapes and device
    print(dataset.spectrograms_array.shape)
    print(dataset.label_array.shape)
    print(dataset.spectrograms_array.device)

    # Test the __get_item__ function
    utterance, utterance_labels = dataset.__get_item__([0])
    print(utterance)
    print(utterance_labels)
    print("---%s seconds ---" % (time.time() - start_time))


def pad_with(tensor, k, dim, device):
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
            npad = (0, 0, k, longest_sequence - array.shape[0])
        if dim == 2:
            npad = (k, longest_sequence - len(array))
        torch_array = torch.tensor(array, device=device)
        padded_array = torch.nn.functional.pad(torch_array,
                                               npad,
                                               mode='constant',
                                               value=0)
        new_tensor.append(padded_array)
    new_tensor = torch.nn.utils.rnn.pad_sequence(new_tensor)
    return new_tensor.transpose(0, 1)


# Inherit Dataset from torch.utils.data
class MelSpectrogramDataset(torch.utils.data.Dataset):
    """Mel Spectrograms dataset."""
    def __init__(self, dataset_file: str, labels_file: str, hyperp_K: int,
                 transform: transforms, device: str):
        """
        Args:
            dataset_file (string): Path to npy file with mel Spectrograms
            labels_file (string): Path to npy file with labels
            hyperp_K (integer): Hyperparameter value K noting prev/post # of context vectors
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.hyperp_K = hyperp_K
        self.device = device
        self.spectrograms_array = np.load(
            dataset_file, allow_pickle=True)  # Load from npy file
        self.spectrograms_array = pad_with(self.spectrograms_array,
                                           self.hyperp_K, 3,
                                           self.device)  # Pad with zeros

        self.label_array = np.load(labels_file,
                                   allow_pickle=True)  # Load from npy file
        self.label_array = pad_with(self.label_array, self.hyperp_K, 2,
                                    self.device)  # Pad with zeros
        self.vector_length = len(self.label_array[0])

        self.transform = transform

    def __len__(self):
        """Returns the length of the dataset
        """
        return len(self.label_array)

    def __getitem__(self, idx):
        """Retrives sample(s) from the dataset
        Args:
            idx (int, tensor): Index of sample(s)
        """
        # Convert index tensor to a list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract example and corresponding label
        example = self.spectrograms_array[idx]
        label = self.label_array[idx]
        return example, label


if __name__ == "__main__":
    main()
