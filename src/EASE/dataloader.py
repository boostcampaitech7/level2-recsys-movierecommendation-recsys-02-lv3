import numpy as np
from scipy import sparse

class EASEDataset:
    '''
    Sparse Matrix Dataset for training, validation, and test data
    '''
    def __init__(self, args, data, datatype='train'):
        self.args = args
        self.data = data
        self.datatype = datatype
        self.n_items = len(data['unique_sid'])
        self.data = self._load_data(self.data, self.datatype)

    def _load_data(self, data, datatype='train'):
        if datatype == 'train':
            tp = data['train_data']
        elif datatype == 'validation':
            tp = data['validation_data']
        elif datatype == 'test':
            tp = data['test_data']
        else:
            tp = data['total_data']

        n_users = tp['uid'].max() + 1
        rows, cols = tp['uid'], tp['sid']

        csr_data = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            dtype='float64',
            shape=(n_users, self.n_items)
        )

        return csr_data
    