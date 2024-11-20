from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch


class DeepFM_DataLoader(Dataset):
    def __init__(self, args, data, datatype="train"):
        self.args = args
        self.data = data
        self.datatype = datatype
        self.device = args.device
        self.dense_cols = ["year"]  # scaling으로 처리할 연속형 변수
        self.scaler = MinMaxScaler().fit(data["total_df"][self.dense_cols])
        self.num_genres = data["total_df"].explode("genre")["genre"].nunique()

        # 데이터셋 설정
        if self.datatype == "train":
            self.dataset = data["train_df"]
        elif self.datatype == "valid":
            self.dataset = data["valid_df"]
        elif self.datatype == "test":
            self.dataset = data["test_df"]
        elif self.datatype == "total":
            self.dataset = data["total_df"]
        else:
            raise ValueError(
                "Invalid mode. Choose from 'train', 'valid', 'test', 'total'."
            )

        self.dataset[self.dense_cols] = self.scaler.transform(
            self.dataset[self.dense_cols]
        )
        self.dataset["genre"] = self.dataset["genre"].apply(
            lambda x: self.genre_to_multi_hot(x)
        )

    # 멀티핫 인코딩 함수 (라벨 인코딩된 장르를 이용)
    def genre_to_multi_hot(self, genre_list):
        genre_vector = np.zeros(self.num_genres)  # 전체 장르 수만큼 0으로 초기화
        for genre in genre_list:
            genre_vector[genre] = 1  # 해당 장르 인덱스에 1을 할당
        return genre_vector

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        user = torch.tensor(row["user"], dtype=torch.long)
        item = torch.tensor(row["item"], dtype=torch.long)
        genre = torch.tensor(row["genre"], dtype=torch.float)
        director = torch.tensor(row["director"], dtype=torch.long)
        year = torch.tensor(row["year"], dtype=torch.float)
        interaction = torch.tensor(row["interaction"], dtype=torch.float)

        return (user, item, genre, director, year), interaction
