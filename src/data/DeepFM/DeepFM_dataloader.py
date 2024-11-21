from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch


class DeepFMDataset(Dataset):
    def __init__(self, args, data, datatype="train"):
        self.args = args
        self.data = data
        self.datatype = datatype
        self.device = args.device
        self.dense_cols = ["year"]  # scaling으로 처리할 연속형 변수
        self.scaler = MinMaxScaler().fit(data["total_df"][self.dense_cols])

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
            lambda x: torch.tensor(x, dtype=torch.long).to(self.device)
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        # 최대 장르 개수 설정
        max_genre_length = 5

        # 장르가 5개 이상이면 자르고, 5개 미만이면 패딩
        genre = row["genre"][:max_genre_length]  # 최대 길이로 자르기

        # 패딩 적용 (길이가 5보다 작으면 0으로 패딩)
        if len(genre) < max_genre_length:
            genre = torch.nn.functional.pad(
                genre, (0, max_genre_length - len(genre)), value=0
            )
        user = torch.tensor(row["user"], dtype=torch.long).to(self.device)
        item = torch.tensor(row["item"], dtype=torch.long).to(self.device)
        director = torch.tensor(row["director"], dtype=torch.long).to(self.device)
        year = torch.tensor(row["year"], dtype=torch.float).to(self.device)
        interaction = torch.tensor(row["interaction"], dtype=torch.float).to(
            self.device
        )

        return (user, item, genre, director, year), interaction
