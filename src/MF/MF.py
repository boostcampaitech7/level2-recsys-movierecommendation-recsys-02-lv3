import torch
import os
import numpy as np
from tqdm import tqdm
from util import evaluate_recall


def loss(rating_mat, X, Y):
    """
    MF 모델은 학습 과정에서 loss 함수를 사용하지 않음
    학습 과정의 모니터링 용도로만 사용
    """
    pred = torch.matmul(X, Y.T)  # 예측 행렬
    mask = rating_mat > 0  # 실제 값이 있는 부분만 계산
    return torch.mean(torch.pow(rating_mat[mask] - pred[mask], 2))  # MSE 계산


def als(
    rating_mat: torch.Tensor,
    answers: list,
    alpha=1,
    epochs=20,
    l1=0.1,
    feature_dim=64,
    device="cuda",
    inference=False,
    tfidf_matrix=None,
    tfidf_weight=1.0,
):
    with torch.no_grad():
        C = rating_mat * alpha + 1
        P = rating_mat

        user_size = rating_mat.size(0)
        item_size = rating_mat.size(1)

        X = torch.rand((user_size, feature_dim), dtype=torch.float32).to(
            device
        )  # User feature matrix
        Y = torch.rand((item_size, feature_dim), dtype=torch.float32).to(
            device
        )  # Item feature matrix

        I = torch.eye(feature_dim, dtype=torch.float32).to(device)
        lI = l1 * I

        pred_rating_mat = torch.matmul(X, Y.T)
        pred_rating_mat[rating_mat > 0] = 0.0
        _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)
        preds = recs.cpu().numpy()

        print("random init.")
        print(f"loss: {loss(rating_mat, X, Y)}")
        if not inference:
            evaluate_recall(answers, preds)

        if tfidf_matrix is not None:
            tfidf_matrix_tensor = torch.tensor(
                tfidf_matrix.toarray(), dtype=torch.float32
            ).to(Y.device)

        for epoch in range(epochs):
            for u in tqdm(range(user_size)):
                left = torch.matmul(Y.T, torch.diag(C[u])) @ Y + lI
                right = torch.matmul(Y.T, torch.diag(C[u])) @ P[u]
                X[u] = torch.linalg.solve(left, right)

            for i in tqdm(range(item_size)):
                if tfidf_matrix is not None:
                    # TF-IDF 정규화 가중치 추가
                    tfidf_vector = tfidf_matrix_tensor[
                        i
                    ]  # TF-IDF 행렬에서 해당 아이템의 벡터
                    left = (
                        torch.matmul(X.T, torch.diag(C[:, i])) @ X
                        + lI
                        + tfidf_weight * torch.eye(feature_dim).to(device)
                    )
                    right = (
                        torch.matmul(X.T, torch.diag(C[:, i])) @ P[:, i]
                        + tfidf_weight * tfidf_vector
                    )
                else:
                    left = X.T.mul(C[:, i]).matmul(X) + lI
                    right = X.T.mul(C[:, i]).matmul(P[:, i])
                Y[i] = torch.linalg.solve(left, right)

            pred_rating_mat = torch.matmul(X, Y.T)
            pred_rating_mat[rating_mat > 0] = 0.0
            _, recs = torch.sort(pred_rating_mat, dim=-1, descending=True)
            preds = recs.cpu().numpy()

            print("epoch", epoch + 1)
            print(f"loss: {loss(rating_mat, X, Y)}")

    return preds
