import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, args, data):
        super(DeepFM, self).__init__()

        self.num_users = data["num_users"]
        self.num_items = data["num_items"]
        self.num_genres = data["num_genres"]
        self.num_directors = data["num_directors"]
        self.embedding_dim = args.model_args[args.model].embed_dim
        self.mlp_dims = args.model_args[args.model].mlp_dims
        self.drop_rate = args.model_args[args.model].drop_rate
        self.device = args.device

        # FM 컴포넌트의 상수 bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,))).to(self.device)  # 상수 bias
        self.fc = nn.Linear(self.embedding_dim * 4 + 1, 1)

        # 임베딩 층 설정
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.genre_embedding = nn.Embedding(self.num_genres, self.embedding_dim)
        self.director_embedding = nn.Embedding(self.num_directors, self.embedding_dim)

        # MLP 계층 설정
        mlp_layers = []
        input_dim = self.embedding_dim * 4 + 1
        for i, dim in enumerate(self.mlp_dims):
            if i == 0:
                mlp_layers.append(nn.Linear(input_dim, dim))
            else:
                mlp_layers.append(
                    nn.Linear(self.mlp_dims[i - 1], dim)
                )  # Linear 층 추가
            mlp_layers.append(nn.ReLU(True))  # 활성화 함수
            mlp_layers.append(nn.Dropout(self.drop_rate))  # 드롭아웃
        mlp_layers.append(nn.Linear(self.mlp_dims[-1], 1))  # 출력 계층
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        """
        FM Component: Second-order interaction term 계산
        """
        # FM의 첫 번째 차수 계산 (Bias + 첫 번째 차수)
        fm_y = self.bias + torch.sum(self.fc(x), dim=1).view(-1, 1)

        # (∑x_i)^2 계산
        square_of_sum = (torch.sum(x, dim=1) ** 2).unsqueeze(1)

        # ∑(x_i^2) 계산
        sum_of_square = torch.sum(x**2, dim=1).unsqueeze(1)

        fm_y += 0.5 * torch.sum(
            square_of_sum - sum_of_square, dim=1, keepdim=True
        )  # FM의 2차 상호작용

        return fm_y

    def mlp(self, x):
        """
        Deep Component (MLP): 모든 임베딩을 이용한 깊은 신경망 처리
        """
        mlp_y = self.mlp_layers(x)
        return mlp_y

    def forward(self, x):
        """
        전체 모델의 forward 함수
        """
        user, item, genre, director, year = x
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        embed_director = self.director_embedding(director)
        embed_genre = self.genre_embedding(genre)
        embed_genre = embed_genre.mean(dim=1)

        embed_x = torch.cat(
            [embed_user, embed_item, embed_genre, embed_director, year.view(-1, 1)],
            dim=1,
        )

        # FM 컴포넌트
        fm_y = self.fm(embed_x).view(-1, 1)

        # Deep Component (MLP)
        mlp_y = self.mlp(embed_x).view(-1, 1)

        # FM과 MLP의 출력 합산 후 Sigmoid 활성화
        y = torch.sigmoid(fm_y + mlp_y)
        return y
