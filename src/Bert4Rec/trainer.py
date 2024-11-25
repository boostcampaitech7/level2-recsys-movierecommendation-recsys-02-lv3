import os
import torch
from tqdm import tqdm
from utils import EarlyStopping

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

    patience = 5  # 성능 개선이 없을 경우 몇 번의 epoch 후 학습 중단
    best_loss = float('inf')  # 초기 최적 Loss는 무한대로 설정
    counter = 0  # Early Stopping을 위한 카운터

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_dir, "best_model.pt")

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
            tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}')

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch:3d} | Average Train Loss: {avg_loss:.5f}")

        # Early Stopping Logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            print(f"Train Loss improved to {best_loss:.5f}. Saving model...")
            torch.save(model.state_dict(), model_save_path)  # save
        else:
            counter += 1
            print(f"No improvement in Train Loss. Early stopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered!")
            break
    