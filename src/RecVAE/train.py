import time
import os
from tqdm import tqdm
from copy import deepcopy
from importlib import import_module

import numpy as np
import torch
from torch import optim

import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.metrics import Recall_at_k_batch

def run(model, opts, dataloader, n_epochs, beta, gamma, device, dropout_rate):
    model.train()
    for epoch in range(n_epochs):
        for idx, batch in enumerate(dataloader):
            data = batch.unsqueeze(0).to(device)

            for opt in opts:
                opt.zero_grad()

            _, loss = model(data, calculate_loss=True, beta=beta, gamma=gamma, dropout_rate = dropout_rate)
            loss.backward()

            for opt in opts:
                opt.step()


def train(args, model, train_dataset, valid_dataset, logger, setting):
    
    # Optimizer and Scheduler 설정
    decoder_params = set(model.decoder.parameters())
    encoder_params = set(model.encoder.parameters())
    optimizer_encoder = optim.Adam(encoder_params, lr=args.optimizer.args.lr)
    optimizer_decoder = optim.Adam(decoder_params, lr=args.optimizer.args.lr)
                        
    # learning parameter 설정
    learning_kwargs = {
        'model':model,
        'beta': None,
        'gamma': 1,
        'device': 'cuda'
    }

    # loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False)
    
    train_scores, valid_scores = [], []
    best_r20 = 0

    for epoch in range(1,args.train.epochs+1):
        
        start_time = time.time()
            
        if args.train.not_alternating:
            run(opts=[optimizer_encoder, optimizer_decoder], dataloader=train_loader, n_epochs=1, dropout_rate=0.5, **learning_kwargs)
        else:
            run(opts=[optimizer_encoder], dataloader=train_loader, n_epochs=3, dropout_rate=0.5, **learning_kwargs)
            model.update_prior()
            run(opts=[optimizer_decoder], dataloader=train_loader, n_epochs=1, dropout_rate=0, **learning_kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time

        train_scores.append(
            evaluate(args, model, train_loader)[0]
        )
        
        valid_scores.append(
            evaluate(args, model, valid_loader)[0]
        )
        
        if valid_scores[-1] > best_r20:
            best_r20 = valid_scores[-1]
            os.makedirs(args.train.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt')
            print(f"🧹 checkpoint updated at epoch {epoch}, recall {best_r20}")
        
        print((f'epoch {epoch} | elapsed time {elapsed_time} | valid recall@20: {valid_scores[-1]:.4f} |' +
               f'best recall@20: {best_r20} | train recall@20: {train_scores[-1]:.4f}'))

    return model

def evaluate(args, model, dataloader):
    model.eval()
    metrics = [{'metric': 'recall', 'k': 20}, {'metric': 'recall', 'k': 50}]
  
    for m in metrics:
        m['score'] = []

    for _, data in enumerate(dataloader):  
        data_tr, data_te = data[0], data[1]
        # 배치 크기와 형태 확인

        items_pred = model(data_tr, calculate_loss=False).cpu().detach().numpy()
        
        if not(data_tr is data_te):
            items_pred[data_tr.cpu().detach().numpy().nonzero()] = -np.inf
        
        for m in metrics:
            m['score'].append(Recall_at_k_batch(items_pred.reshape(1,-1), data_te.reshape(1,-1).cpu().detach().numpy(), k=m['k']))
        
    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()
        
    return [x['score'] for x in metrics]
    


def test(args, model, dataset, setting, checkpoint = True):
    
    if checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        if args.train.save_best_model:
            model_path = f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt'
        else:
            # best가 아닐 경우 마지막 에폭으로 테스트하도록 함
            model_path = f'{args.train.save_dir.checkpoint}/{setting.save_time}_{args.model}_e{args.train.epochs-1:02d}.pt'
        model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Run on test data
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.dataloader.batch_size, shuffle=False)

    r = evaluate(args, model, test_loader)
    print('=' * 89)
    print('| End of training | r20 {:4.2f} | r50 {:4.2f}'.format(r[0],r[1]))
    print('=' * 89)
    
    return model