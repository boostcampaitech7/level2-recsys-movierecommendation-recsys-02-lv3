from scipy.sparse import csr_matrix
from scipy import sparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from copy import deepcopy
import torch

class EASER:
    def __init__(self, args):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

        model_args = args.model_args['EASER'] 
        self.threshold =model_args['threshold']
        self.lambdaBB = model_args['lambdaBB']
        self.lambdaCC = model_args['lambdaCC']
        self.rho = model_args['rho']
        self.epochs = model_args['epochs']

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, 'user'])
        items = self.item_enc.fit_transform(df.loc[:, 'item'])
        
        return users, items
    
    def create_list_feature_pairs(self, XtX):
        AA = np.triu(np.abs(XtX))
        AA[ np.diag_indices(AA.shape[0]) ]=0.0
        ii_pairs = np.where((AA > self.threshold) == True)
        return ii_pairs
    
    def create_matrix_Z(self, ii_pairs, X):
        MM = np.zeros( (len(ii_pairs[0]), X.shape[1]),    dtype=np.float64)
        MM[np.arange(MM.shape[0]) , ii_pairs[0]   ]=1.0
        MM[np.arange(MM.shape[0]) , ii_pairs[1]   ]=1.0
        CCmask = 1.0-MM
        MM = sparse.csc_matrix(MM.T)
        Z=  X * MM
        Z= (Z == 2.0 )
        Z=Z*1.0
        return Z, CCmask
    
    def train_higher(self, XtX, XtXdiag, ZtZ, ZtZdiag, CCmask, ZtX):
        ii_diag=np.diag_indices(XtX.shape[0])
        XtX[ii_diag] = XtXdiag + self.lambdaBB
        print('XtX updated with lambdaBB')

        PP = np.linalg.inv(XtX)
        print('PP (XtX inverse) calculated')

        ii_diag_ZZ=np.diag_indices(ZtZ.shape[0])
        ZtZ[ii_diag_ZZ] = ZtZdiag + self.lambdaCC + self.rho
        print('ZtZ updated with lambdaCC and rho')

        QQ=np.linalg.inv(ZtZ)
        print('QQ (ZtZ inverse) calculated')


        CC = np.zeros( (ZtZ.shape[0], XtX.shape[0]),dtype=np.float64 )
        DD = np.zeros( (ZtZ.shape[0], XtX.shape[0]),dtype=np.float64 )
        UU = np.zeros( (ZtZ.shape[0], XtX.shape[0]),dtype=np.float64 )

        for _ in tqdm(range(self.epochs)):
            # learn BB
            XtX[ii_diag] = XtXdiag
            BB= PP.dot(XtX-ZtX.T.dot(CC))
            gamma = np.diag(BB) / np.diag(PP)
            BB-= PP * gamma
            # learn CC
            CC= QQ.dot(ZtX-ZtX.dot(BB) + self.rho * (DD-UU))
            # learn DD
            DD = CC * CCmask 
            # DD = np.maximum(0.0, DD) # if you want to enforce non-negative parameters
            # learn UU (is Gamma in paper)
            UU+= CC-DD
    
        return BB, DD
        
    def fit(self, df, predict_save_path=None):       
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
        )
        X = csr_matrix((values, (users, items)))
        self.X = X     
                     
        print(' --- init --- ')
        XtX = (X.transpose() * X).toarray()
        print('XtX calculated, shape:', XtX.shape)

        XtXdiag = deepcopy(np.diag(XtX))
        ii_pairs = self.create_list_feature_pairs(XtX)
        print('ii_pairs created, length:', len(ii_pairs[0]))

        Z, CCmask = self.create_matrix_Z(ii_pairs, X)
        print('Z matrix created, shape:', Z.shape)

        ZtZ = (Z.transpose() * Z).toarray()
        print('ZtZ matrix created, shape:')

        ZtZdiag = deepcopy(np.diag(ZtZ))
        print('ZtZdiag matrix created')

        ZtX = (Z.transpose() * X).toarray()
        
        
        print(' --- iteration start.--- ')
        BB, CC = self.train_higher(XtX, XtXdiag, ZtZ, ZtZdiag, CCmask, ZtX)
        print(' --- iteration end.--- ')

        self.pred = torch.from_numpy(X.toarray().dot(BB) + Z.toarray().dot(CC))

        if predict_save_path:  # 경로가 제공된 경우에만 저장
            np.save(predict_save_path, self.pred.numpy())  # self.pred 저장
            print(f"Predictions saved as numpy array at {predict_save_path}")
        
        return self.pred 
     
    
    def predict(self, train, users, items, k):
        items = self.item_enc.transform(items)
        train_df = train.loc[train.user.isin(users)]
        train_df['ci'] = self.item_enc.transform(train_df.item)
        train_df['cu'] = self.user_enc.transform(train_df.user)
        groupby_user = train_df.groupby('user')
        
        user_preds = pd.DataFrame()
        for user, group in tqdm(groupby_user):
            watched = set(group['ci'])  # The movie users watched
            candidates = [item for item in items if item not in watched]
            
            predict_user = group.cu.iloc[0]
            pred = np.take(self.pred[predict_user,:], candidates)
            res = np.argpartition(pred, -k)[-k:] # top-K
            r = pd.DataFrame(
                {
                    "user": [user] * len(res),
                    "item": np.take(candidates, res),
                    "score": np.take(pred, res),
                }
            ).sort_values('score', ascending=False)
            user_preds = pd.concat([user_preds,r])

        user_preds['item'] = self.item_enc.inverse_transform(user_preds['item'])

      
        # Generate dictionaries for user and item mappings
        user_index_to_id = {index: user_id for index, user_id in enumerate(self.user_enc.classes_)}
        item_index_to_id = {index: item_id for index, item_id in enumerate(self.item_enc.classes_)}  

        return user_preds, user_index_to_id, item_index_to_id
        