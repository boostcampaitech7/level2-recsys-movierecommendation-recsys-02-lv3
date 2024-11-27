import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import glob

def _merge_data(args):
    data_path = args.dataset.data_path

    genre_data = pd.read_csv(os.path.join(data_path, 'genres.tsv'), sep = '\t')
    raw_data = pd.read_csv(os.path.join(data_path, 'train_ratings.csv'))

    item_ids = raw_data['item'].unique()
    user_ids = raw_data['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)

    # user, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

    genre_names = genre_data.genre.unique()
    genre2idx = {genre : idx + 1 for idx, genre in enumerate(genre_names)}

    merged_data = raw_data.merge(genre_data, how = 'left', on = 'item')
    merged_data = pd.merge(merged_data, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    merged_data = pd.merge(merged_data, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    merged_data['genre_idx'] = merged_data['genre'].apply(lambda x: genre2idx[x])
    merged_data.sort_values(['user_idx', 'time'], inplace=True)
    del merged_data['item'], merged_data['user'], merged_data['genre']

    num_genres = len(set(merged_data.genre_idx))
    
    return merged_data, item_ids, user_ids, num_item, num_user, num_genres, item2idx, user2idx, genre2idx

def _split_train_test(args, merged_data):
    save_path = args.dataset.save_path
    files = glob.glob(os.path.join(save_path, '*.csv'))
    if save_path:
        train_data = pd.read_csv(os.path.join(save_path, 'merged_train_data.csv'))
        test_data = pd.read_csv(os.path.join(save_path, 'merged_test_data.csv'))
        print("Load train, test data in existed folder")

    else:
        merged_train_data_list = []
        merged_test_data_list = []

        for idx, tmp in tqdm(merged_data.groupby('user_idx')):
            # 중복 제거 및 순서 유지
            last_10_item = list(dict.fromkeys(tmp.item_idx))[-10:]
            
            # Boolean indexing으로 train/test 나누기
            mask = tmp.item_idx.isin(last_10_item)
            tmp_train = tmp[~mask]
            tmp_test = tmp[mask][['user_idx', 'item_idx']].drop_duplicates()

            # 리스트에 바로 추가 (DataFrame 연산 최소화)
            merged_train_data_list.append(tmp_train)
            merged_test_data_list.append(tmp_test)

        # 최종적으로 한 번에 concat
        train_data = pd.concat(merged_train_data_list, ignore_index=True)
        test_data = pd.concat(merged_test_data_list, ignore_index=True)


    return train_data, test_data


def _split_train_valid(train_data):
    train_dic = defaultdict(list)
    train_df = {}
    valid_df = {}

    for u, i in zip(train_data["user_idx"], train_data["item_idx"]):
        train_dic[u].append(i)

    for user in train_dic:
        user_list = list(dict.fromkeys(train_dic[user]))
        train_df[user] = user_list[:-1]
        valid_df[user] = [user_list[-1]]

    return train_df, valid_df

def get_total_data(merged_data):
    total_dic = defaultdict(list)
    total_df = {}

    for u, i in zip(merged_data["user_idx"], merged_data["item_idx"]):
        total_dic[u].append(i)

    for user in total_dic:
        user_list = list(dict.fromkeys(total_dic[user]))
        total_df[user] = user_list

    return total_df


def get_item_genre_dic(train_data):
    item_genre_dic = {}
    for idx, tmp in tqdm(train_data.groupby('item_idx')):
        item_genre_dic[idx] = list(set(tmp.genre_idx))

    item_genre_dic[6808] = [0,0,0]

    return item_genre_dic

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

    merged_data, item_ids, user_ids, num_item, num_user, num_genres, item2idx, user2idx, genre2idx = _merge_data(args)

    train_data, test_data = _split_train_test(args, merged_data)

    train_df, valid_df = _split_train_valid(train_data)

    item_genre_dic = get_item_genre_dic(train_data)

    total_df = get_total_data(merged_data)

    data = {
        'merged_data': merged_data,
        'item_ids': item_ids,
        'user_ids': user_ids,
        'num_item': num_item,
        'num_user': num_user,
        'num_genres': num_genres,
        'item2idx': item2idx,
        'user2idx': user2idx,
        'genre2idx': genre2idx,
        'train_data': train_data,
        'test_data': test_data,
        'train_df': train_df,
        'valid_df': valid_df,
        'item_genre_dic': item_genre_dic,
        'total_df': total_df
    }

    return data