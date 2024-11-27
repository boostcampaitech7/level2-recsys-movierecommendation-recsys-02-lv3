import torch.nn as nn
import torch

from tqdm import tqdm
import numpy as np
import glob
import os
import wandb
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.util import transform_df_to_dict
from metrics import recall_at_k
from model import ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward, BERT4RecBlock, BERT4Rec


def train(args, model, train_dataloader, valid_dataloader, logger):
    output_dir = args.train.ckpt_dir
    os.makedirs(output_dir, exist_ok=True)

    best_val_acc = 0
    best_val_loss = np.inf

    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)

    print (f"Start of TRAINING")

    for epoch in range(args.train.epochs):
        #-- training
        loss_sum = 0
            
        tqdm_bar = tqdm(train_dataloader)
            
        for idx, (log_seqs, labels, log_genres) in enumerate(tqdm_bar):
            logits = model(log_seqs, log_genres)
                
            # size matching
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1).to(args.device)      
                
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss_sum += loss
            loss.backward()
            optimizer.step()
                
            tqdm_bar.set_description(f'Epoch: {epoch + 1:3d}| Step: {idx:3d}| Train loss: {loss:.5f}')
            
        train_loss_avg = loss_sum / len(train_dataloader)

        #-- validataion
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            masked_cnt = 0
            correct_cnt = 0

            for _log_seqs, _labels, _log_genres in valid_dataloader:
                _logits = model(_log_seqs, _log_genres)

                y_hat = _logits[:,:].argsort()[:,:,-1].view(-1)

                # size matching
                _logits = _logits.view(-1, _logits.size(-1))   # [6400, 6808]
                _labels = _labels.view(-1).to(args.device)         # 6400

                _loss = criterion(_logits, _labels)
                            
                correct_cnt += torch.sum((_labels == y_hat) & (_labels != 0))
                masked_cnt += _labels.count_nonzero()
                valid_loss += _loss
                        
            valid_loss_avg = valid_loss / len(valid_dataloader)
            valid_acc = correct_cnt / masked_cnt

            # wandb logging
            if args.wandb:
                wandb.log({
                    "Validation Loss": valid_loss_avg,
                    "Validation Accuracy": valid_acc,
                    "Epoch": epoch,
                })
            
            ##### 이 부분 best model 선택할 때 acc로 하는 건 어떰
            if best_val_acc < valid_acc:
                print(f"New best model for val acc : {valid_acc * 100:.5f}! saving the best model..")
                torch.save(model, f"{output_dir}/best.pt")
                best_val_loss = valid_loss_avg
                best_val_acc = valid_acc
                stop_counter = 0
                
            torch.save(model, f"{output_dir}/last.pt")
                
            print(
                f"Validation Accuracy : {valid_acc:4.2%}, Validation Loss: {valid_loss_avg:.5f} || "
                f"Best Accuracy: {best_val_acc:4.2%}, Best Loss: {best_val_loss:.5f}"
            )
            print('=' * 120)
        
        logger.log(epoch=epoch, train_loss=train_loss_avg, valid_loss=valid_loss_avg, valid_r10 = 0)

    logger.close()

    return model

