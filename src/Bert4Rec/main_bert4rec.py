import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from src.Bert4Rec.model import BERT4Rec
from src.Bert4Rec.dataloader import get_dataloader
from src.Bert4Rec.trainer import train_bert4rec
from src.Bert4Rec.utils import preprocess_data

def random_neg(l, r, s):
    # log에 존재하는 아이템과 겹치지 않도록 sampling
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def main():

    # 하이퍼파라미터 설정
    hidden_units = 50
    num_heads = 1
    num_layers = 2
    dropout_rate = 0.5
    device = 'cuda'
    max_len = 50
    num_workers = 1

    lr = 0.001
    batch_size = 128
    num_epochs = 5  # 테스트용으로 작은 값
    mask_prob = 0.15

    # 데이터 전처리
    data_path = "data/train/train_ratings.csv"
    # user_train, num_user, num_item = preprocess_data(data_path, batch_size)
    user_train, user_valid, num_user, num_item = preprocess_data(data_path, batch_size)

    # 데이터 로더
    dataloader = get_dataloader(user_train, num_user, num_item, max_len, mask_prob, batch_size)

    # 모델 초기화
    torch.cuda.empty_cache()
    model = BERT4Rec(num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device)
    model.to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # label이 0인 경우 무시
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 학습 시작
    train_bert4rec(model, dataloader, criterion, optimizer, device, num_epochs)

    # 학습 후 평가 코드 추가
    print("Evaluating model...")
    model.eval()

    NDCG = 0.0 # NDCG@10
    HIT = 0.0 # HIT@10

    num_item_sample = 100
    num_user_sample = 1000
    users = np.random.randint(0, num_user, num_user_sample) # 샘플 사용자 선택

    for u in users:
        seq = (user_train[u] + [num_item + 1])[-max_len:]
        rated = set(user_train[u] + user_valid[u])
        item_idx = [user_valid[u][0]] + [random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)]

        with torch.no_grad():
            predictions = -model(np.array([seq]))
            predictions = predictions[0][-1][item_idx]  # 샘플링된 아이템들에 대한 예측값
            rank = predictions.argsort().argsort()[0].item()

        if rank < 10: # Top-10에 정답이 포함된 경우
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1

    print(f'NDCG@10: {NDCG / num_user_sample} | HIT@10: {HIT / num_user_sample}')



if __name__ == "__main__":
    main()
