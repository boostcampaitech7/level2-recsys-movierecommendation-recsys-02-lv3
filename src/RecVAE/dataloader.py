import torch
import numpy as np
from scipy import sparse
from torch.utils.data import Dataset

class RecVAEDataset(Dataset):
    '''
    Sparse Matrix Dataset for training, validation, and test data
    '''
    def __init__(self, args, data, datatype='train'):
        self.args = args
        self.data = data
        self.datatype = datatype
        self.device = args.device
        self.n_items = len(data['unique_sid'])

        if self.datatype == 'train':
            self.data = self._load_train_data(self.data)
        elif self.datatype in ['validation', 'test']:
            self.data_tr, self.data_te = self._load_tr_te_data(self.data, self.datatype)
        elif self.datatype == 'total':
            self.data = self._load_total_data(self.data)
        else:
            raise ValueError("datatype should be in [train, validation, test, total]")

    @property
    def total_data(self):
        if self.datatype == 'total':
            return self.data
        else:
            raise ValueError("total_data property is only available for 'total' datatype.")


    def __len__(self):
        if self.datatype in ['validation', 'test']:
            return self.data_tr.shape[0]
        else:
            return self.data.shape[0]
        
    def __getitem__(self, idx):
        print(f"get item {idx}")
        if self.datatype in ['validation', 'test']:
            return self._sparse_to_tensor(self.data_tr[idx]), self._sparse_to_tensor(self.data_te[idx])
        else:
            return self._sparse_to_tensor(self.data[idx])


    # csr_matrix to tensor
    def _sparse_to_tensor(self, sparse_matrix):
        '''
        CSR matrix -> Tensor 변환 함수
        '''
        if isinstance(sparse_matrix, torch.Tensor):
            return sparse_matrix.to(self.device)
        elif isinstance(sparse_matrix, sparse.csr_matrix):
            dense_matrix = sparse_matrix.toarray()
            return torch.FloatTensor(dense_matrix).to(self.device)
        
        else:
            raise ValueError("Unsupported matrix type")

        
        
    def _load_total_data(self, data):
        tp = data['total_data']
        n_users = tp['uid'].max() + 1
        n_items = len(tp.sid.unique())
        rows, cols = tp['uid'], tp['sid']

        total_data = sparse.csr_matrix((np.ones_like(rows),
                                        (rows, cols)), dtype='float64',
                                       shape=(n_users, n_items))
        
        total_data = self._sparse_to_tensor(total_data)

        return total_data


    def _load_train_data(self, data):
        tp = data['train_data']
        n_users = tp['uid'].max() + 1
        rows, cols = tp['uid'], tp['sid']

        train_data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            dtype='float64',
            shape=(n_users, self.n_items)
        )

        train_data = self._sparse_to_tensor(train_data)
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
        return data_tr, data_te