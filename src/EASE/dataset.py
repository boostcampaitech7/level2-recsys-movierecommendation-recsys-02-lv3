import numpy as np
import pandas as pd
import argparse
import yaml
from scipy import sparse


##### multi-vae dataset을 위한 preprocessing 함수들 #####

def _get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    return playcount_groupbyid.size()


def _numerize(tp, user2idx, item2idx):
    uid = tp['user'].apply(lambda x: user2idx[x])
    sid = tp['item'].apply(lambda x: item2idx[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


##### data split function

def _split_data(raw_data, tr_users, vd_users, te_users, unique_uid):
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

    # output
    data = {
        'train_data': _numerize(train_plays.copy(), user2idx, item2idx),
        'validation_data': _numerize(vad_plays.copy(), user2idx, item2idx),
        'test_data': _numerize(test_plays.copy(), user2idx, item2idx),
        'total_data' : _numerize(raw_data.copy(), user2idx, item2idx),
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
    
    data = _split_data(train_df, tr_users, vd_users, te_users, unique_uid)
    
    return data