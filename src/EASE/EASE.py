import numpy as np


class EASE:
    def __init__(self, _lambda):
        self.B = None
        self._lambda = _lambda    
    def train(self, X):
        """
        Trains the EASE model.

        Parameters:
        - X (scipy.sparse.csr_matrix): Interaction matrix of shape (user_num, item_num).

        """
        G = X.T @ X  # G = X'X
        diag_indices = np.arange(G.shape[0])
        G[diag_indices, diag_indices] += self._lambda  # X'X + λI
        P = np.linalg.inv(G)  # P = (X'X + λI)^(-1)
        
        self.B = P / -np.diag(P)  # - P_{ij} / P_{jj} if i ≠ j
        min_dim = min(*self.B.shape)  
        self.B[range(min_dim), range(min_dim)] = 0  # 대각행렬 원소만 0으로 만들어주기 위해
    
    def forward(self, user_row):
        return user_row @ self.B