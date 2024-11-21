import torch
from tqdm import tqdm

# for epoch in range(1, num_epochs + 1):
#     tbar = tqdm(data_loader)
#     for step, (log_seqs, labels) in enumerate(tbar):
#         logits = model(log_seqs)

#         # size matching
#         logits = logits.view(-1, logits.size(-1))
#         labels = labels.view(-1).to(device)

#         optimizer.zero_grad()
#         loss = criterion(logits, labels)
#         loss.backward()
#         optimizer.step()
 
#         tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}')


def train_bert4rec(model, dataloader, criterion, optimizer, device, num_epochs):
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
    model.to(device)  # 모델을 지정한 디바이스로 이동
    model.train()  # 모델을 학습 모드로 설정

    for epoch in range(1, num_epochs + 1):
        tbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        
        for step, (log_seqs, labels) in enumerate(tbar):
            # 데이터를 디바이스로 이동
            log_seqs, labels = log_seqs.to(device), labels.to(device)
            
            # 모델 예측
            logits = model(log_seqs)

            # 텐서 크기 맞추기
            logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_len, num_classes]
            labels = labels.view(-1)  # [batch_size * seq_len]

            # 손실 계산 및 역전파
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # 진행 상황 업데이트
            tbar.set_description(f"Epoch: {epoch} | Step: {step} | Train loss: {loss.item():.5f}")

