import os
import pandas as pd
from collections import defaultdict
import numpy as np
import torch


def preprocess_data(data_path, batch_size): # for train

    # Load data 
    df = pd.read_csv(data_path)  
    item_ids = df['item'].unique() 
    user_ids = df['user'].unique() 
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // batch_size 

    # User, item to index
    item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)  # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)  # user re-indexing (0~num_user-1)

    # Dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user']

    # Train set, valid set 생성
    users = defaultdict(list)  # defaultdict은 dictionary의 key가 없을 때 default 값을 value로 반환
    user_train = {}
    user_valid = {}

    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    return user_train, user_valid, num_user, num_item



def preprocess_all_data(data_path): # for submission 

    # Load data
    df = pd.read_csv(data_path)
    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)

    # User, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids)) + 1, index=item_ids)  # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids)  # user re-indexing (0~num_user-1)
    idx2user = pd.Series(data=user_ids, index=user2idx.values)  # re-indexed user → original user
    idx2item = pd.Series(data=item_ids, index=item2idx.values)  # re-indexed item → original item

    # Dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user']

    # Make data
    users = defaultdict(list)  # defaultdict: dictionary의 key가 없을 때 default 값을 value로 반환
    user_train = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)
        
    for user in list(users.keys()):
        user_train[user] = users[user][:]

    return user_train, num_user, num_item, idx2user, idx2item



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score



def make_submission(predicted_user, predicted_item, idx2user, idx2item, file_name = 'submission.csv'):

    # Get current dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)

    # DataFrame
    submission_df = pd.DataFrame({
        'user': predicted_user,
        'item': predicted_item
    })

    # Get back to Original Index 
    submission_df['user'] = submission_df['user'].map(idx2user)
    submission_df['item'] = submission_df['item'].map(idx2item)

    # Save
    submission_df.to_csv(file_path, index=False)
    print("Submission saved!")



def evaluate_model(model, user_train, user_valid, num_user, num_item, max_len, num_item_sample=100, num_user_sample=1000, device='cuda'):
    print("Evaluating model...")
    model.eval()

    NDCG = 0.0  # NDCG@10
    HIT = 0.0   # HIT@10

    # Select sample users
    users = np.random.randint(0, num_user, num_user_sample)

    def random_neg(l, r, s): # log에 존재하는 아이템과 겹치지 않도록 sampling
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    for u in users:
        seq = (user_train[u] + [num_item + 1])[-max_len:] #  add input token for next prediction
        true_items = user_valid[u]  # answer movie list
        rated = set(user_train[u] + user_valid[u])
        
        # Item sampling for evaluation
        item_idx = [user_valid[u][0]] + [random_neg(1, num_item + 1, rated) for _ in range(num_item_sample)]

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

    print(f'NDCG@10: {NDCG / num_user_sample} | HIT@10: {HIT / num_user_sample}')
    return NDCG / num_user_sample, HIT / num_user_sample

