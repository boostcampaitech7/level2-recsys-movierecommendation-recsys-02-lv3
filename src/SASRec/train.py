import torch
import numpy as np
from tqdm import tqdm
from SASRec import SASRec
from dataloader import SASRecDataset, sample_batch, random_neg

def train(config):
    dataset = SASRecDataset(config['train']['data_path'])
    
    model = SASRec(
        num_user=dataset.num_user,
        num_item=dataset.num_item,
        hidden_units=config['model']['hidden_units'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        maxlen=config['model']['max_len'],
        dropout_rate=config['model']['dropout_rate'],
        device=config['train']['device']
    )
    model.to(config['train']['device'])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    num_batch = dataset.num_user // config['train']['batch_size']
    
    for epoch in range(1, config['train']['num_epochs'] + 1):
        model.train()
        total_loss = 0
        
        tbar = tqdm(range(num_batch))
        for step in tbar:
            user, seq, pos, neg = sample_batch(
                dataset.user_train, 
                dataset.num_user,
                dataset.num_item,
                config['train']['batch_size'],
                config['model']['max_len']
            )
            
            pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=config['train']['device'])
            neg_labels = torch.zeros(neg_logits.shape, device=config['train']['device'])
            
            optimizer.zero_grad()
            indices = np.where(pos != 0)
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}')
            
    return model, dataset

def evaluate(model, dataset, config):
    model.eval()
    
    NDCG = 0.0  
    HIT = 0.0   
    
    users = np.random.randint(0, dataset.num_user, config['eval']['num_user_sample'])
    
    for u in users:
        seq = dataset.user_train[u][-config['model']['max_len']:]
        rated = set(dataset.user_train[u] + dataset.user_valid[u])
        item_idx = dataset.user_valid[u] + [random_neg(1, dataset.num_item + 1, rated) 
                                          for _ in range(config['eval']['num_item_sample'])]
        
        with torch.no_grad():
            predictions = -model.predict(np.array([seq]), np.array(item_idx))
            predictions = predictions[0]
            rank = predictions.argsort().argsort()[0].item()
            
        if rank < 10 :
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            
    metrics = {
        'NDCG@10': NDCG/config['eval']['num_user_sample'],  
        'HIT@10': HIT/config['eval']['num_user_sample']
    }
    
    return metrics 