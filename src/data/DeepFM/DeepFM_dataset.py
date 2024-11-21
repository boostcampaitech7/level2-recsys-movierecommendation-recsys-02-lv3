import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import re


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
        unique_values = set(data.explode(type)[type])
        encoding_dict = {value: i for i, value in enumerate(unique_values)}

        # 리스트 내부 각 요소를 인코딩
        data[type] = data[type].map(lambda x: [encoding_dict[val] for val in x])

    else:  # 컬럼 값이 일반 문자열(혹은 단일 값)인 경우
        encoding_dict = {value: i for i, value in enumerate(set(data["director"]))}
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

    # item_metadata에서 genre는 그대로 리스트로 두고 다른 메타데이터는 중복 제거
    item_metadata = (
        data.groupby("item")
        .agg(
            {
                "director": "first",  # 첫 번째 감독 정보 사용
                "year": "first",  # 첫 번째 년도 정보 사용
                "genre": "first",  # genre는 리스트 그대로 유지
            }
        )
        .reset_index()
    )

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
        # 메타데이터와 결합
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
        idx2users = {i: users for users, i in users_dict.items()}
        data["user"] = data["user"].map(lambda x: users_dict[x])

    if len(items) - 1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        idx2items = {i: items for items, i in items_dict.items()}
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
        if len(group) > 1:  # 최소 2개의 상호작용이 있어야 valid, test로 분할 가능
            train_data.append(group.iloc[:-2])  # 마지막 두 개를 제외한 나머지
            valid_data.append(group.iloc[-2:-1])  # 마지막에서 두 번째를 valid
            test_data.append(group.iloc[-1:])  # 마지막 상호작용을 test
        else:
            # 상호작용이 하나만 있는 경우는 모두 train으로만 사용
            train_data.append(group)

    train_df = pd.concat(train_data)
    valid_df = (
        pd.concat(valid_data) if valid_data else pd.DataFrame(columns=data.columns)
    )
    test_df = pd.concat(test_data) if test_data else pd.DataFrame(columns=data.columns)

    # interaction이 0인 데이터는 train에만 포함
    data_negative = data[data["interaction"] == 0]
    train_df = pd.concat([train_df, data_negative])

    return train_df, valid_df, test_df


def handle_missing_value(data):
    """
    data : 영화 메타 데이터(director, writer, year, title)를 포함한 데이터 프레임
    director, writer, year에 대한 결측치 처리 후 반환
    """
    # 'director' 컬럼의 결측치는 'unknown'으로 채우기
    data["director"] = data["director"].fillna("unknown")

    # 'writer' 컬럼의 결측치는 'unknown'으로 채우기
    data["writer"] = data["writer"].fillna("unknown")

    # 'year' 결측치는 'title' 컬럼의 끝부분에 있는 (year) 부분을 추출해서 채우기
    def extract_year_from_title(title):
        """
        제목에서 (year) 형태의 개봉 연도를 추출하는 함수
        """
        match = re.search(r"\((\d{4})\)", title)
        if match:
            return float(match.group(1))  # 'year'는 float64 타입이므로 변환
        return None

    data["year"] = data["year"].fillna(data["title"].apply(extract_year_from_title))

    return data


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
    data = handle_missing_value(result_df).drop(columns=["writer", "title", "time"])

    # 'genre', 'director' 컬럼 임베딩 전 label encoding
    data = encoding(data, "genre")
    data = encoding(data, "director")

    # user, item 컬럼 zero-based index mapping
    data, idx2user, idx2item = zero_based_index_mapping(data)

    # interaction 컬럼 추가 (상호작용 여부 1 or 0)
    data["interaction"] = 1.0

    # negative sampling
    # data = negative_sampling(data, args.model_args[args.model].num_negative)

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
        "num_users": total_df["user"].nunique(),
        "num_items": total_df["item"].nunique(),
        "num_genres": total_df.explode("genre")["genre"].nunique(),
        "num_directors": total_df["director"].nunique(),
    }

    return data
