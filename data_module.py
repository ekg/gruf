import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

class Enwik8DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=4, seq_len=512, data_path="./data/enwik8.gz"):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_path = data_path

    def setup(self, stage=None):
        # Load and prepare the Enwik8 dataset
        with gzip.open(self.data_path) as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            np_train, np_valid = np.split(data, [int(90e6)])
            self.data_train = torch.from_numpy(np_train)
            self.data_val = torch.from_numpy(np_valid)
        
        self.train_dataset = TextSamplerDataset(self.data_train, self.seq_len)
        self.val_dataset = TextSamplerDataset(self.data_val, self.seq_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
