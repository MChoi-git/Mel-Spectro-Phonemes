import torch
import torch.nn as nn
import torch.nn.functional as F


MODELS = {}

# Register model decorator
def register(func=None, *, name=None):
    def wrapper(func):
        MODELS[func.__name__ if name is None else name] = func
        return func
    if func is None:
        return wrapper
    return wrapper(func)

def get_model(name, *args):
    func = MODELS.get(name, None)
    return func(*args) if func is not None else func



# Define neural net
@register
class SimpleNet(nn.Module):
    def __init__(self, data_size):
        super().__init__()
        self.fc1 = nn.Linear(40, 120)
        self.fc2 = nn.Linear(120, 200)
        self.fc3 = nn.Linear(200, 71)

    def forward(self, input_example):
        input_example = self.fc1(input_example)
        input_example = F.relu(input_example)
        input_example = self.fc2(input_example)
        input_example = F.relu(input_example)
        input_example = self.fc3(input_example)
        return input_example
