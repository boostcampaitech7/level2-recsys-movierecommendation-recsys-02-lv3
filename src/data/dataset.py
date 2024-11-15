import numpy as np
import pandas as pd
import regex
import argparse
import torch
import yaml
from torch.utils.data import TensorDataset, DataLoader
import src.data.preprocessing as pre
import os



def _split_data(raw_data, tr_users, vd_users, te_users):
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
    # 1. 훈련 데이터 추출
    train_plays = raw_data.loc[raw_data['user'].isin(tr_users)]

    # 2. 아이템 ID 추출 및 매핑
    unique_sid = pd.unique(train_plays['item'])
    show2id = {sid: i for i, sid in enumerate(unique_sid)}
    profile2id = {pid: i for i, pid in enumerate(pd.unique(raw_data['user']))}
    id2show = {i: sid for i, sid in enumerate(unique_sid)}
    id2profile = {i: pid for i, pid in enumerate(pd.unique(raw_data['user']))}
    

    # 3. Validation, Test 데이터 전처리
    vad_plays = raw_data.loc[raw_data['user'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item'].isin(unique_sid)]

    test_plays = raw_data.loc[raw_data['user'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['item'].isin(unique_sid)]
    
    vad_plays_tr, vad_plays_te = pre.split_train_test_proportion(vad_plays)
    test_plays_tr, test_plays_te = pre.split_train_test_proportion(test_plays)

    # 4. 데이터 넘버링 (유저 ID와 아이템 ID를 숫자로 매핑)
    train_data = pre.numerize(train_plays.copy(), profile2id, show2id)
    vad_data_tr = pre.numerize(vad_plays_tr.copy(), profile2id, show2id)
    vad_data_te = pre.numerize(vad_plays_te.copy(), profile2id, show2id)
    test_data_tr = pre.numerize(test_plays_tr.copy(), profile2id, show2id)
    test_data_te = pre.numerize(test_plays_te.copy(), profile2id, show2id)

    # 5. 결과 반환
    data = {
        'train_data': train_data,
        'validation_tr': vad_data_tr,
        'validation_te': vad_data_te,
        'test_tr': test_data_tr,
        'test_te': test_data_te,
        'unique_sid': unique_sid,
        'id2show' : id2show,
        'id2profile' : id2profile
    }
    
    return data




def _split_user_indices(user_activity, n_heldout_users=3000, seed=0):
    """
    유저 인덱스를 셔플한 후, Train/Validation/Test 데이터셋으로 분할하는 함수입니다.

    Parameters
    ----------
    user_activity : pandas.DataFrame
        'user' 열이 포함된 데이터프레임 (유저 활동 정보).
    n_heldout_users : int, optional
        Validation과 Test 셋에 할당할 유저 수 (기본값은 3000).
    seed : int, optional
        난수 시드 값.

    Returns
    -------
    tr_users : numpy.ndarray
        Train 셋에 포함된 유저 인덱스 배열.
    vd_users : numpy.ndarray
        Validation 셋에 포함된 유저 인덱스 배열.
    te_users : numpy.ndarray
        Test 셋에 포함된 유저 인덱스 배열.
    """
    # 1. 유저 ID를 unique한 값으로 추출
    unique_uid = user_activity['user'].unique()
    
    # 2. 유저 ID 셔플
    np.random.seed(seed)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]
    
    # 3. Train/Validation/Test 유저 인덱스 분할
    n_users = unique_uid.size
    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2):(n_users - n_heldout_users)]
    te_users = unique_uid[(n_users - n_heldout_users):]
    
    return tr_users, vd_users, te_users


def data_load(args):
    """
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

    raw_data, user_activity, item_popularity = pre.filter_triplets(train_df, min_uc=5, min_sc=0)

    data = {
        'data' : raw_data,
        'user_activity' : user_activity,
        'item_popularity' : item_popularity
    }
    
    raw_data = data['data'] 
    tr_users, vd_users, te_users = _split_user_indices(data['user_activity'])
    
    data = _split_data(raw_data, tr_users, vd_users, te_users)
    
    
    return data


if __name__ == '__main__':
    
    with open('./choi/level2-recsys-movierecommendation-recsys-02-lv3/configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                        default=config['dataset']['data_path'],
                        help='Movielens dataset location')
    
    # 인자 파싱
    args = parser.parse_args()
    
    data = data_load(args)
    
    
    #데이터 셋 확인
    print(data['train_data'])
    print(data['validation_tr'])
    print(data['validation_te'])
    # print(test_data_tr)
    # print(test_data_te)