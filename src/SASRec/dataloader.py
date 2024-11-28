import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from collections import defaultdict

def random_neg(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_batch(user_train, num_user, num_item, batch_size, max_len):
    def sample():
        user = np.random.randint(num_user)
        seq = np.zeros([max_len], dtype=np.int32)
        pos = np.zeros([max_len], dtype=np.int32)
        neg = np.zeros([max_len], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = max_len - 1

        train_item = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neg(1, num_item + 1, train_item)
            nxt = i
            idx -= 1
            if idx == -1: break
        return (user, seq, pos, neg)
    
    user, seq, pos, neg = zip(*[sample() for _ in range(batch_size)])
    return np.array(user), np.array(seq), np.array(pos), np.array(neg)

class SASRecDataset:
    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        
        # Create mapping dictionaries
        self.user2idx = {user:idx for idx, user in enumerate(df['user'].unique())}
        self.idx2user = {idx:user for idx, user in enumerate(df['user'].unique())}
        self.item2idx = {item:(idx+1) for idx, item in enumerate(df['item'].unique())}
        self.idx2item = {(idx+1):item for idx, item in enumerate(df['item'].unique())}
        
        self.num_user = len(self.user2idx)
        self.num_item = len(self.item2idx)
        
        # Convert IDs to indices
        df['user_idx'] = df['user'].map(self.user2idx)
        df['item_idx'] = df['item'].map(self.item2idx)
        df.sort_values(['user_idx', 'time'], inplace=True)
        
        # train set, valid set 생성
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        
        for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
            users[u].append(i)

        for user in users:
            user_train[user] = users[user][:-1]
            user_valid[user] = [users[user][-1]]
            
        self.user_train = user_train
        self.user_valid = user_valid