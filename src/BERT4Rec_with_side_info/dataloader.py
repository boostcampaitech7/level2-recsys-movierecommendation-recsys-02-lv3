import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class BERT4Rec_with_side_infoDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, hidden_units, max_len, mask_prob, item_metadata):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.item_metadata = item_metadata
        self.hidden_units = hidden_units

    def __len__(self):
        return self.num_user

    def __getitem__(self, user):
        item_list = self.user_train[user]
        tokens = []
        labels = []
        
        genres = []  # 아이템 별 장르 평균 임베딩 모음 
        
        for item in item_list:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.num_item + 1)  # mask_index
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, self.num_item + 1))  # 랜덤 아이템
                else:
                    tokens.append(item)
                labels.append(item)
                
                # 마스킹일 경우 genre를 0으로 처리
                genres.append([0] * 3)
            else:
                tokens.append(item)
                labels.append(0)
            
                # 메타데이터 처리 (장르 및 작가 임베딩)
                genre_list = self.item_metadata[item][:3]
                padding_len = 3 - len(genre_list)
                padded_genre_list = [genre_list[0]] * padding_len + genre_list
                genres.append(padded_genre_list)
         
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        genres = genres[-self.max_len:]
        
        # zero padding
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        #genres = [0] * mask_len + genres
        genres = [[0] * 3 for _ in range(mask_len)] + genres
        
        # 메타데이터 임베딩 합치기
        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(genres)
