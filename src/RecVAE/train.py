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

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def train(args, model, train_dataset, valid_dataset, logger, setting):
    
    # Optimizer and Scheduler 설정
    decoder_params = set(model.decoder.parameters())
    encoder_params = set(model.encoder.parameters())
    optimizer_encoder = optim.Adam(encoder_params, lr=args.optimizer.args.lr)
    optimizer_decoder = optim.Adam(decoder_params, lr=args.optimizer.args.lr)
                        
    # learning parameter 설정
    learning_kwargs = {
        'model': model,
        'beta': None,
        'gamma': 0.004,
        'device': 'cuda'
    }
    
    train_scores, valid_scores = [], []
    best_r10 = 0

    for epoch in range(1,args.train.epochs+1):
        
        start_time = time.time()
            
        if args.train.not_alternating:
            run(args=args.dataloader, opts=[optimizer_encoder, optimizer_decoder], data=train_dataset, n_epochs=1, dropout_rate=0.5, **learning_kwargs)
        else:
            run(args=args.dataloader, opts=[optimizer_encoder], data=train_dataset, n_epochs=3, dropout_rate=0.5, **learning_kwargs)
            model.update_prior()
            run(args=args.dataloader, opts=[optimizer_decoder], data=train_dataset, n_epochs=1, dropout_rate=0, **learning_kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time

        train_scores.append(
            evaluate(args.dataloader, model, data_in=train_dataset, data_out=train_dataset)[0]
        )
        
        valid_scores.append(
            evaluate(args.dataloader, model, data_in=valid_dataset.data_tr, data_out=valid_dataset.data_te)[0]
        )
        
        if valid_scores[-1] > best_r10:
            best_r10 = valid_scores[-1]
            os.makedirs(args.train.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt')
            print(f"\n🧹 checkpoint updated at epoch {epoch}, recall {best_r10}")
        
        print((f'epoch {epoch} | elapsed time {elapsed_time} | valid recall@10: {valid_scores[-1]:.4f} |' +
               f'best recall@10: {best_r10} | train recall@20: {train_scores[-1]:.4f}'))

    return model


def run(args, model, opts, data, n_epochs, beta, gamma, device, dropout_rate):
    model.train()
    for epoch in range(n_epochs):
        for batch in generate(batch_size=args.batch_size, device=device, data_in=data, shuffle=args.shuffle):
            ratings = batch.get_ratings_to_dev()
            for optimizer in opts:
                optimizer.zero_grad()
                
            _, loss = model(ratings, beta=beta, gamma=gamma, dropout_rate=dropout_rate)
            loss.backward()
            
            for optimizer in opts:
                optimizer.step()


def evaluate(args, model, data_in, data_out):
    model.eval()
    metrics = [{'metric': 'recall', 'k': 10}, {'metric': 'recall', 'k': 50}]
  
    for m in metrics:
        m['score'] = []

    for batch in generate(batch_size=args.batch_size,
                          device=args.device,
                          data_in=data_in,
                          data_out=data_out,
                          samples_perc_per_epoch=1
                         ):
        
        ratings_in = batch.get_ratings_to_dev()
        ratings_out = batch.get_ratings(is_out=True)
    
        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
        
        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf
        
        for m in metrics:
            m['score'].append(Recall_at_k_batch(ratings_pred, ratings_out, k=m['k']))
        
    for m in metrics:
        m['score'] = np.concatenate(m['score']).mean()
        
    return [x['score'] for x in metrics]
    


def test(args, model, test_data,  setting, checkpoint=None):
    data_in = test_data.data_tr
    data_out = test_data.data_te

    if checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    else:
        ### checkpoint = True && best_model test
        if args.train.save_best_model:
            model_path = f'{args.train.ckpt_dir}/{setting.save_time}_{args.model}_best.pt'
            model.load_state_dict(torch.load(model_path, weights_only=True))
    
    # Run on test data
    r = evaluate(args.dataloader, model, data_in=data_in, data_out=data_out)
    print('=' * 89)
    print('| End of training | r10 {:4.2f} | r50 {:4.2f}'.format(r[0],r[1]))
    print('=' * 89)
    
    return model