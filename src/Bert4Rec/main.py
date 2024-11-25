import os 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
from model import BERT4Rec
from dataloader import get_dataloader
from trainer import train_bert4rec
from utils import preprocess_data, preprocess_all_data, make_submission, evaluate_model
import pandas as pd
from collections import defaultdict
from inference import generate_recommendations
from torch.utils.data import DataLoader
from dataset import SeqDataset

def random_neg(l, r, s): # log에 존재하는 아이템과 겹치지 않도록 sampling
    t = np.random.randint(l, r) 
    while t in s: 
        t = np.random.randint(l, r)
    return t



def main():

    # Model setting
    max_len = 100
    hidden_units = 50
    num_heads = 1
    num_layers = 2
    dropout_rate= 0.3
    num_workers = 1
    device = 'cuda'

    # Training setting
    lr = 0.001
    batch_size = 128
    num_epochs = 200
    mask_prob = 0.15 


    # Data preprocessing
    data_path = 'data/train/train_ratings.csv'
    user_train, user_valid, num_user, num_item = preprocess_data(data_path, batch_size)

    # Dataloader 
    dataloader = get_dataloader(user_train, num_user, num_item, max_len, mask_prob, batch_size)

    # Model initialization
    torch.cuda.empty_cache()
    model = BERT4Rec(num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    train_bert4rec(model, dataloader, criterion, optimizer, device, num_epochs)

    # Evaluation
    evaluate_model(model, user_train, user_valid, num_user, num_item, max_len, device=device)

    # Save model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_path)

    # Load saved model
    model = BERT4Rec(num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device).to(device)  # 모델 초기화
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 전환

    user_train, num_user, num_item, idx2user, idx2item = preprocess_all_data(data_path)

    users = list(user_train.keys())  # 전체 사용자 리스트
    output_path = 'src/Bert4Rec/submission_df.csv'
    generate_recommendations(model, users, user_train, num_user, num_item, max_len, idx2user, idx2item, output_path, device)




if __name__ == "__main__":
    main()
