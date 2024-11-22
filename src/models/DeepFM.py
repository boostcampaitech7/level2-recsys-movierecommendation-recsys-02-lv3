import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, args, field_dims):
        super(DeepFM, self).__init__()

        self.field_dims = field_dims
        self.input_dims = sum(self.field_dims)
        self.embedding_dim = args.model_args[args.model].embed_dim
        self.mlp_dims = args.model_args[args.model].mlp_dims
        self.drop_rate = args.model_args[args.model].drop_rate
        self.batch_size = args.dataloader.batch_size
        self.device = args.device

        # FM 컴포넌트의 상수 bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))  # 상수 bias
        self.fc = nn.Embedding(self.input_dims, 1)

        self.embedding = nn.Embedding(self.input_dims, self.embedding_dim)

        # MLP 계층 설정
        mlp_layers = []
        for i, dim in enumerate(self.mlp_dims):
            if i == 0:
                mlp_layers.append(
                    nn.Linear(len(self.field_dims) * self.embedding_dim, dim)
                )
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
        embed_x = self.embedding(x)

        # FM의 첫 번째 차수 계산 (Bias + 첫 번째 차수)
        fm_y = self.bias + torch.sum(self.fc(x), dim=1)

        # (∑x_i)^2 계산
        square_of_sum = torch.sum(embed_x, dim=1) ** 2

        # ∑(x_i^2) 계산
        sum_of_square = torch.sum(embed_x**2, dim=1)

        fm_y += 0.5 * torch.sum(
            square_of_sum - sum_of_square, dim=1, keepdim=True
        )  # FM의 2차 상호작용
        return fm_y

    def mlp(self, x):
        """
        Deep Component (MLP): 모든 임베딩을 이용한 깊은 신경망 처리
        """
        embed_x = self.embedding(x)
        inputs = embed_x.view(-1, len(self.field_dims) * self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        # fm component
        fm_y = self.fm(x).squeeze(1)

        # deep component
        mlp_y = self.mlp(x).squeeze(1)

        y = torch.sigmoid(fm_y + mlp_y)

        return y
