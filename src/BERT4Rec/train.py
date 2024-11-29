import os
import numpy as np
import torch
from tqdm import tqdm
from utils import EarlyStopping


def train(model, dataloader, criterion, optimizer, device, num_epochs, model_path):
    """
    BERT4Rec 모델 훈련 함수

    Args:
        model: 훈련할 BERT4Rec 모델
        dataloader: PyTorch DataLoader (훈련 데이터)
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: GPU/CPU 디바이스
        num_epochs: 총 훈련 에포크 수
    """

    patience = 5  # 성능 개선이 없을 경우 몇 번의 epoch 후 학습 중단
    best_loss = float("inf")  # 초기 최적 Loss는 무한대로 설정
    counter = 0  # Early Stopping을 위한 카운터

    # 최적 모델 저장 경로
    best_model_path = os.path.join(model_path, "BERT4Rec_best.pt")
    final_model_path = os.path.join(model_path, "BERT4Rec_final.pt")

    for epoch in range(1, num_epochs + 1):
        # Training Loop
        model.train()
        tbar = tqdm(dataloader)
        total_loss = 0

        for step, (log_seqs, labels) in enumerate(tbar):
            logits = model(log_seqs)

            # size matching
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1).to(device)

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tbar.set_description(
                f"Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}"
            )

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{num_epochs} - Average Train Loss: {avg_loss:.5f}")

        # Early Stopping Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch} with loss {avg_loss:.5f}")
        else:
            counter += 1
            print(f"No improvement. Early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered!")
            break

    # Save final model
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")


def evaluate(
    model,
    user_train,
    user_valid,
    num_user,
    num_item,
    max_len,
    num_item_sample=100,
    num_user_sample=1000,
    device="cuda",
):
    print("Evaluating model...")
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0  # HIT@10

    # Select sample users
    users = np.random.randint(0, num_user, num_user_sample)

    def random_neg(l, r, s):  # log에 존재하는 아이템과 겹치지 않도록 sampling
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    for u in users:
        seq = (user_train[u] + [num_item + 1])[
            -max_len:
        ]  #  add input token for next prediction
        true_items = user_valid[u]  # answer movie list
        rated = set(user_train[u] + user_valid[u])

        # Item sampling for evaluation
        item_idx = [user_valid[u][0]] + [
            random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)
        ]

        with torch.no_grad():
            predictions = -model(torch.tensor([seq], dtype=torch.long).to(device))
            predictions = predictions[0, -1, item_idx]  # sampling
            rank = predictions.argsort().argsort()[0].item()
            top_10_indices = predictions.argsort()[:10]  # top 10 indexes

            # top 10 items
            recommended_items = [item_idx[i] for i in top_10_indices]

        # Calculate evaluation metrics
        if rank < 10:  # if prediction is correct
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    print(f"NDCG@10: {NDCG / num_user_sample} | HIT@10: {HIT / num_user_sample}")
    return NDCG / num_user_sample, HIT / num_user_sample
