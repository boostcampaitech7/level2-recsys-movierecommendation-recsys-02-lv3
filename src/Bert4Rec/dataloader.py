import torch
from torch.utils.data import DataLoader
from dataset import SeqDataset

def get_dataloader(user_train, num_user, num_item, max_len, mask_prob, batch_size):
    seq_dataset = SeqDataset(user_train, num_user, num_item, max_len, mask_prob)
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader