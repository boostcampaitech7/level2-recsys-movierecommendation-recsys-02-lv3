import numpy as np
import pandas as pd
import argparse
import yaml
from scipy import sparse


##### multi-vae dataset을 위한 preprocessing 함수들 #####

def _get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    return playcount_groupbyid.size()


def _split_train_test_proportion(args, data):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(args.seed)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(args.dataset.test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)


    return pd.concat(tr_list), pd.concat(te_list)


def _numerize(tp, user2idx, item2idx):
    uid = tp['user'].apply(lambda x: user2idx[x])
    sid = tp['item'].apply(lambda x: item2idx[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])



def tensor_to_csr(tensor):
    tensor = tensor.cpu().numpy()

    rows, cols = np.where(tensor != 0)
    values = tensor[rows, cols]
    num_rows, num_cols = tensor.shape
    csr_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(num_rows, num_cols))
    
    return csr_matrix



##### data split function

def _split_data(args, raw_data, tr_users, vd_users, te_users, unique_uid):
    """
    훈련, 검증, 테스트 데이터를 전처리하고 분할하는 함수입니다.
    
    Parameters
    ----------
    raw_data : pandas.DataFrame
        전체 유저-아이템 상호작용 데이터.
    tr_users : numpy.ndarray
        훈련 데이터에 포함될 유저 인덱스 배열.
    vd_users : numpy.ndarray
        검증 데이터에 포함될 유저 인덱스 배열.
    te_users : numpy.ndarray
        테스트 데이터에 포함될 유저 인덱스 배열.

    Returns
    -------
    train_data : pandas.DataFrame
        훈련 데이터.
    vad_data_tr : pandas.DataFrame
        검증 데이터의 Train 부분.
    vad_data_te : pandas.DataFrame
        검증 데이터의 Test 부분.
    test_data_tr : pandas.DataFrame
        테스트 데이터의 Train 부분.
    test_data_te : pandas.DataFrame
        테스트 데이터의 Test 부분.
    """
    train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]

    unique_sid = pd.unique(train_plays['item'])
    item2idx = {item: idx for (idx, item) in enumerate(unique_sid)}
    user2idx = {user: idx for (idx, user) in enumerate(unique_uid)}
    id2item = {idx: item for (idx, item) in enumerate(unique_sid)}
    id2user = {idx: user for (idx, user) in enumerate(unique_uid)}
    
    
    vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]

    test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]
    
    vad_plays_tr, vad_plays_te = _split_train_test_proportion(args, vad_plays)
    test_plays_tr, test_plays_te = _split_train_test_proportion(args, test_plays)

    # output
    data = {
        'train_data': _numerize(train_plays.copy(), user2idx, item2idx),
        'validation_tr': _numerize(vad_plays_tr.copy(), user2idx, item2idx),
        'validation_te': _numerize(vad_plays_te.copy(), user2idx, item2idx),
        'test_tr': _numerize(test_plays_tr.copy(), user2idx, item2idx),
        'test_te': _numerize(test_plays_te.copy(), user2idx, item2idx),
        'total_data' : _numerize(raw_data, user2idx, item2idx),
        'unique_sid': unique_sid,
        'id2item' : id2item,
        'id2user' : id2user
    }
    
    return data


def data_load(args):
    """
    
    유저 인덱스를 셔플한 후, Train/Validation/Test 데이터셋으로 분할합니다.
    이후에 data를 load해주는 함수입니다.
    
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 데이터와 user, item 정보가 담긴 사전 형식의 데이터를 반환합니다.
    """
    
    train_df = pd.read_csv(args.dataset.data_path + 'train_ratings.csv', header=0)

    user_activity = _get_count(train_df, 'user')
    unique_uid = user_activity['user'].unique()
    
    np.random.seed(args.seed)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
        
    n_users = unique_uid.size
    n_heldout_users = int(n_users * args.dataset.ratio)

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2):(n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    
    data = _split_data(args, train_df, tr_users, vd_users, te_users, unique_uid)
    
    return data