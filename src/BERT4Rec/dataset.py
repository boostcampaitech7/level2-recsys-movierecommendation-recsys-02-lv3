import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class SeqDataset(Dataset):
    '''
    사용자-아이템 상호작용 시퀀스를 위한 데이터셋

    SeqDataset:
    사용자 훈련 데이터를 처리하여 상호작용 시퀀스를 생성하고, 마스킹과 랜덤 치환을 적용하며,
    모델 입력용으로 패딩된 시퀀스와 해당 라벨을 반환하는 클래스.
    '''
    def __init__(self, user_train, num_user, num_item, max_len, mask_prob):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): # 특정 유저가 본 영화 시퀀스 속 각 영화에 대해서, 
        # iterator를 구동할 때 사용
        seq = self.user_train[user]
        tokens = []
        labels = []

        for s in seq: # 특정 유저가 본 영화 시퀀스 속 각 영화에 대해서, 
            prob = np.random.random() # prob값을 randomize  한 뒤 
            if prob < self.mask_prob: # 조건 만족하면 
                prob /= self.mask_prob # 지정 

                # BERT 학습 - 마스킹
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                # 랜덤 치환
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, self.num_item+1))  # item random sampling
                # 원래 값 유지
                else:
                    tokens.append(s)
                labels.append(s)  # 학습에 사용
            else:
                tokens.append(s)
                labels.append(0)  # 학습에 사용 X, trivial
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        # zero padding
        tokens = [0] * mask_len + tokens # 리턴값 1. tokens: 모델이 학습에 사용할 입력 시퀀스로, 마스킹 및 치환이 적용되어있다
        labels = [0] * mask_len + labels # 리턴값 2. labels: 마스킹된 위치에서만 원래 값을 포함하는 라벨 시퀀스이다. 마스킹 되지 않은 곳은 전부 0
        return torch.LongTensor(tokens), torch.LongTensor(labels)



def preprocess_data(data_path, batch_size):

    # Load data 
    df = pd.read_csv(data_path)  
    item_ids = df['item'].unique() 
    user_ids = df['user'].unique() 
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // batch_size 

    # User, item to index
    item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)  # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)  # user re-indexing (0~num_user-1)

    # Dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user']

    # Train set, valid set 생성
    users = defaultdict(list)  # defaultdict은 dictionary의 key가 없을 때 default 값을 value로 반환
    user_train = {}
    user_valid = {}

    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    return user_train, user_valid, num_user, num_item



def preprocess_all_data(data_path):

    # Load data
    df = pd.read_csv(data_path)
    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)

    # User, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)  # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)  # user re-indexing (0~num_user-1)
    idx2user = pd.Series(data=user_ids, index=user2idx.values)  # re-indexed user → original user
    idx2item = pd.Series(data=item_ids, index=item2idx.values)  # re-indexed item → original item

    # Dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user']

    # Make data
    users = defaultdict(list)  # defaultdict: dictionary의 key가 없을 때 default 값을 value로 반환
    user_train = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)
        
    for user in list(users.keys()):
        user_train[user] = users[user][:]

    return user_train, num_user, num_item, idx2user, idx2item