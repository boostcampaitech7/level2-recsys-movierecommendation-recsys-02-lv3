import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, args, loader):
        super(DeepFM, self).__init__()

        self.num_users = loader.num_users
        self.num_items = loader.num_items
        self.num_genres = loader.num_genres
        self.num_directors = loader.num_directors
        self.embedding_dim = args[args.model].embed_dim
        self.mlp_dims = args[args.model].mlp_dims
        self.drop_rate = args[args.model].drop_rate

        # FM 컴포넌트의 상수 bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))  # 상수 bias
        self.fc = nn.Embedding(self.embedding_dim * 4, 1)

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
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
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
        user, item, genre, director, year = x
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        embed_director = self.director_embedding(director)

        # genre는 멀티핫 인코딩된 벡터 (1인 인덱스만 임베딩)
        genre_embed = self.genre_embedding(genre)  # genre는 멀티핫 인코딩된 벡터
        nonzero_genre_indices = genre.nonzero(as_tuple=True)[
            1
        ]  # (batch_size, num_genres)에서 1인 인덱스 추출
        selected_genre_embeddings = genre_embed[nonzero_genre_indices]
        genre_avg_embed = selected_genre_embeddings.mean(dim=1)  # 평균

        # FM의 첫 번째 차수 계산 (Bias + 첫 번째 차수)
        fm_y = self.bias + torch.sum(
            self.fc(
                torch.cat(
                    [embed_user, embed_item, genre_avg_embed, embed_director], dim=1
                )
            ),
            dim=1,
        )

        # 2차 상호작용을 위한 계산
        square_of_sum = (
            torch.sum(embed_user + embed_item + embed_director + genre_avg_embed, dim=1)
            ** 2
        )  # (∑x_i)^2
        sum_of_square = torch.sum(
            (embed_user + embed_item + embed_director + genre_avg_embed) ** 2, dim=1
        )  # ∑(x_i^2)
        fm_y += 0.5 * torch.sum(
            square_of_sum - sum_of_square, dim=1, keepdim=True
        )  # FM의 2차 상호작용

        fm_y += year  # 연속형 변수 year는 추가적으로 FM의 예측값에 더해줌

        return fm_y

    def mlp(self, x):
        """
        Deep Component (MLP): 모든 임베딩을 이용한 깊은 신경망 처리
        """
        user, item, genre, director, year = x
        embed_user = self.user_embedding(user)
        embed_item = self.item_embedding(item)
        embed_director = self.director_embedding(director)

        # genre 임베딩을 포함한 입력 벡터 준비
        genre_embed = self.genre_embedding(genre)
        nonzero_genre_indices = genre.nonzero(as_tuple=True)[1]
        selected_genre_embeddings = genre_embed[nonzero_genre_indices]
        genre_avg_embed = selected_genre_embeddings.mean(dim=1)

        inputs = torch.cat(
            [embed_user, embed_item, embed_director, genre_avg_embed, year.view(-1, 1)],
            dim=1,
        )
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        """
        전체 모델의 forward 함수
        """
        # FM 컴포넌트
        fm_y = self.fm(x).squeeze(1)

        # Deep Component (MLP)
        mlp_y = self.mlp(x).squeeze(1)

        # FM과 MLP의 출력 합산 후 Sigmoid 활성화
        y = torch.sigmoid(fm_y + mlp_y)
        return y
