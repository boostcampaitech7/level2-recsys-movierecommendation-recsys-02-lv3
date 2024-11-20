from preprocessing import handle_missing_value
import torch.nn as nn
import pandas as pd
import numpy as np
import tqdm


def encoding(data, type):
    """
    object 타입의 컬럼들을 label encoding으로 변환

    parameters
    ----------
    data : encoding할 type 컬럼이 포함된 데이터 프레임
    type : 컬럼명(ex. 'genre', 'director', 'writer' 등)
    """
    # 리스트인지 확인
    is_list_column = isinstance(data[type].iloc[0], list) if not data.empty else False

    if is_list_column:  # 컬럼 값이 리스트인 경우
        # 유니크한 값 추출
        unique_values = set(val for sublist in data for val in sublist)
        encoding_dict = {value: i for i, value in enumerate(unique_values)}

        # 리스트 내부 각 요소를 인코딩
        data[type] = data[type].map(lambda x: [encoding_dict[val] for val in x])

    else:  # 컬럼 값이 일반 문자열(혹은 단일 값)인 경우
        encoding_dict = {value: i for i, value in enumerate(set(data))}
        data[type] = data[type].map(lambda x: encoding_dict[x])

    return data


def negative_sampling(data, num_negative):
    """
    Parameters
    ----------
    data : 'user', 'item'의 interaction이 포함된 데이터 프레임
    """
    user_group_dfs = list(data.groupby("user")["item"])
    items = set(data["item"])
    item_metadata = data[["item", "genre", "director", "year"]].drop_duplicates()

    print("-----negative sampling-----")
    user_neg_dfs = []
    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)
        i_user_neg_item = np.random.choice(
            list(items - u_items), num_negative, replace=False
        )
        i_user_neg_df = pd.DataFrame(
            {
                "user": [u] * num_negative,
                "item": i_user_neg_item,
                "interaction": [0] * num_negative,
            }
        )
        i_user_neg_df = i_user_neg_df.merge(item_metadata, on="item", how="left")
        user_neg_dfs.append(i_user_neg_df)

    user_neg_dfs = pd.concat(user_neg_dfs, axis=0, sort=False)
    data = pd.concat([data, user_neg_dfs], axis=0, sort=False)

    return data


def zero_based_index_mapping(data):
    """
    0부터 시작하지 않는 id를 가진 컬럼을 0부터 시작하도록 변경
    """
    users = list(set(data.loc[:, "user"]))
    users.sort()
    items = list(set((data.loc[:, "item"])))
    items.sort()

    if len(users) - 1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        idx2users = {i: users[i] for users[i], i in users_dict}
        data["user"] = data["user"].map(lambda x: users_dict[x])

    if len(items) - 1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        idx2items = {i: items[i] for items[i], i in items_dict}
        data["item"] = data["item"].map(lambda x: items_dict[x])

    data = data.sort_values(by=["user"])
    data.reset_index(drop=True, inplace=True)

    return data, idx2users, idx2items


def train_valid_test_split(data):
    """
    data를 train, valid, test로 split
    user 별 마지막 상호작용 두 개 중 하나를 valid, 나머지 하나를 test로 설정
    """
    # interaction이 1인 데이터만 가져와서 valid, test로 분할
    data_positive = data[data["interaction"] == 1.0]

    train_data = []
    valid_data = []
    test_data = []

    for user, group in data_positive.groupby("user"):
        train_data.append(group.iloc[:-2])
        valid_data.append(group.iloc[-2:-1])
        test_data.append(group.iloc[-1:])

    train_df = pd.concat(train_data)
    valid_df = pd.concat(valid_data)
    test_df = pd.concat(test_data)

    # interaction이 0인 데이터는 train에만 포함
    data_negative = data[data["interaction"] == 0]
    train_df = pd.concat([train_df, data_negative])

    return train_df, valid_df, test_df


def data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser

    Returns
    -------
    data : dict
        아래와 같은 데이터들을 딕셔너리 형태로 반환합니다.
        train_df, valid_df, test_df, idx2user, idx2item
    """
    path = args.dataset.data_path
    rating_df = pd.read_csv(path + "train_ratings.csv")
    directors_df = pd.read_csv(path + "directors.tsv", delimiter="\t")
    genres_df = pd.read_csv(path + "genres.tsv", delimiter="\t")
    titles_df = pd.read_csv(path + "titles.tsv", delimiter="\t")
    writers_df = pd.read_csv(path + "writers.tsv", delimiter="\t")
    years_df = pd.read_csv(path + "years.tsv", delimiter="\t")
    result_df = rating_df.copy()

    genres_df = (
        genres_df.groupby("item").agg(genre=("genre", lambda x: list(x))).reset_index()
    )
    writers_df = (
        writers_df.groupby("item")
        .agg(writer=("writer", lambda x: list(x)))
        .reset_index()
    )

    dfs = [directors_df, titles_df, years_df, writers_df, genres_df]

    for df in dfs:
        result_df = pd.merge(result_df, df, on="item", how="left")

    # 결측치 처리 후 사용하지 않을 컬럼 drop
    data = handle_missing_value(result_df).drop(columns=["writer", "title"])

    # 'genre', 'director' 컬럼 임베딩 전 label encoding
    data = encoding(data, "genre")
    data = encoding(data, "director")

    # user, item 컬럼 zero-based index mapping
    data, idx2user, idx2item = zero_based_index_mapping(data)

    # interaction 컬럼 추가 (상호작용 여부 1 or 0)
    data["interaction"] = 1.0

    # negative sampling
    data = negative_sampling(data, args[args.model].num_negative)

    total_df = data

    # train, valid, test split
    train_df, valid_df, test_df = train_valid_test_split(data)

    data = {
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "total_df": total_df,
        "idx2user": idx2user,
        "idx2item": idx2item,
    }

    return data
