import os
import pandas as pd
from scipy import sparse
import numpy as np
import re

def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count

# 특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상)
# 데이터만을 추출할 때 사용하는 함수입니다.
# 현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        itemcount = get_count(tp, 'item')
        tp = tp[tp['item'].isin(itemcount[itemcount['size'] >= min_sc]['item'])]

    if min_uc > 0:
        usercount = get_count(tp, 'user')
        tp = tp[tp['user'].isin(usercount[usercount['size'] >= min_uc]['user'])]

    usercount, itemcount = get_count(tp, 'user'), get_count(tp, 'item')
    return tp, usercount, itemcount

#훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
#100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
#확인하기 위함입니다.
def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby('user')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['user'].apply(lambda x: profile2id[x])
    sid = tp['item'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def preprocessing_time(data, time_drop=True):
    '''
    data : 'time' 컬럼을 가진 데이터 프레임
    time 컬럼을 'time_year', 'month', 'day_of_week', 'hour'로 변경하여 데이터 프레임 반환
    time_drop : 반환 시 time 컬럼을 드랍할 것인지 결정 (default=True)
    '''
    data['time'] = pd.to_datetime(data['time'], unit='s', errors='coerce')
    
    data['time_year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day_of_week'] = data['time'].dt.dayofweek  # 0=월요일, 6=일요일
    data['hour'] = data['time'].dt.hour

    if time_drop:
        data.drop(columns=['time'], inplace=True)

    return data

def handle_missing_value(data):
    '''
    data : 영화 메타 데이터(director, writer, year, title)를 포함한 데이터 프레임
    director, writer, year에 대한 결측치 처리 후 반환
    '''
    # 'director' 컬럼의 결측치는 'unknown'으로 채우기
    data['director'] = data['director'].fillna('unknown')
    
    # 'writer' 컬럼의 결측치는 'unknown'으로 채우기
    data['writer'] = data['writer'].fillna('unknown')
    
    # 'year' 결측치는 'title' 컬럼의 끝부분에 있는 (year) 부분을 추출해서 채우기
    def extract_year_from_title(title):
        '''
        제목에서 (year) 형태의 개봉 연도를 추출하는 함수
        '''
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return float(match.group(1))  # 'year'는 float64 타입이므로 변환
        return None
    
    data['year'] = data['year'].fillna(data['title'].apply(extract_year_from_title))
    
    return data