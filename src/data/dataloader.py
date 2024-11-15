import argparse
import time
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, TensorDataset


class MultiVAE_DataLoader():
    '''
    Load Movielens dataset
    '''
    def __init__(self, args, data):
        self.n_items = len(data['unique_sid'])
        self.data = data
        self.device = args.device
        
    def load_data(self, datatype = 'train', predict = False):
        if datatype == 'train':
            if predict == False:
                return self._load_train_data(self.data, False)
            elif predict == True:
                return self._load_train_data(self.data, True)
        elif datatype == 'validation':
            return self._load_tr_te_data(self.data, datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(self.data, datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def _load_train_data(self, data, type = False):
        tp = data['train_data']
        n_users = tp['uid'].max() + 1
        rows, cols = tp['uid'], tp['sid']

        # CSR matrix 생성
        train_data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)), 
            dtype='float64', 
            shape=(n_users, self.n_items)
        )


        if type == False:
            # Tensor로 변환 후 반환
            train_data_tensor = self._sparse_to_tensor(train_data)
            return train_data_tensor
        elif type == True:
            # 최종 예측의 경우 바로 출력
            return train_data
        
        

    def _load_tr_te_data(self, data, datatype='test'):
        if datatype == 'validation':
            tp_tr = data['validation_tr']
            tp_te = data['validation_te']
        elif datatype == 'test':
            tp_tr = data['test_tr']
            tp_te = data['test_te']

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix(
            (np.ones_like(rows_tr), (rows_tr, cols_tr)), 
            dtype='float64', 
            shape=(end_idx - start_idx + 1, self.n_items)
        )

        data_te = sparse.csr_matrix(
            (np.ones_like(rows_te), (rows_te, cols_te)), 
            dtype='float64', 
            shape=(end_idx - start_idx + 1, self.n_items)
        )

        # Tensor로 변환 후 반환
        data_tr_tensor = self._sparse_to_tensor(data_tr)
        data_te_tensor = self._sparse_to_tensor(data_te)
        
        return data_tr_tensor, data_te_tensor

    def _sparse_to_tensor(self, sparse_matrix):
        '''
        CSR matrix -> Tensor 변환 함수
        '''
        dense_matrix = sparse_matrix.toarray()
        tensor = torch.FloatTensor(dense_matrix).to(self.device)
        return tensor
    
    
 