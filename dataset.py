from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.block_size = block_size
        self.inputs = []
        for i in range(0, len(tokens) - block_size):
            self.inputs.append((tokens[i:i+block_size], tokens[i+1:i+1+block_size]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x, y = self.inputs[idx]
        return torch.tensor(x), torch.tensor(y)
