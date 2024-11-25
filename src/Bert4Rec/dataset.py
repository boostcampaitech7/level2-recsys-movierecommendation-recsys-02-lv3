import torch
import numpy as np
from torch.utils.data import Dataset

class SeqDataset(Dataset):
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