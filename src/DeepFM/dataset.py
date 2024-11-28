import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import torch


def mapping(data):
    """
    데이터프레임의 컬럼값들을 0부터 시작하는 인덱스로 mapping

    parameters
    ----------
    data : 'user', 'item', 'genre', 'director', 'writer', 'title', 'year' 컬럼을 포함하는 데이터 프레임

    returns
    -------
    data : mapping된 데이터 프레임
    field_dims : 각 컬럼들의 고유값 차원 수
    idx2user : mapping된 user를 다시 원래대로 되돌리기 위한 dict
    idx2item : mapping된 item을 다시 원래대로 되돌리기 위한 dict
    idx_dict : 각 컬럼을 mapping하기 위한 dict들
    """
    offset = 0  # mapping 시작 인덱스
    user2idx = {user: idx for idx, user in enumerate(data["user"].unique(), offset)}
    data["user"] = data["user"].map(user2idx)

    offset += len(user2idx)  # 시작 인덱스 갱신
    item2idx = {item: idx for idx, item in enumerate(data["item"].unique(), offset)}
    data["item"] = data["item"].map(item2idx)

    offset += len(item2idx)
    genre2idx = {genre: idx for idx, genre in enumerate(data["genre"].unique(), offset)}
    data["genre"] = data["genre"].map(genre2idx)

    offset += len(genre2idx)
    director2idx = {
        director: idx for idx, director in enumerate(data["director"].unique(), offset)
    }
    data["director"] = data["director"].map(director2idx)

    offset += len(director2idx)
    writer2idx = {
        writer: idx for idx, writer in enumerate(data["writer"].unique(), offset)
    }
    data["writer"] = data["writer"].map(writer2idx)

    offset += len(writer2idx)
    year2idx = {year: idx for idx, year in enumerate(data["year"].unique(), offset)}
    data["year"] = data["year"].map(year2idx)

    field_dims = np.array(
        [
            len(user2idx),
            len(item2idx),
            len(genre2idx),
            len(director2idx),
            len(writer2idx),
            len(year2idx),
        ],
        dtype=np.uint32,
    )

    idx2user = {idx: user for user, idx in user2idx.items()}
    idx2item = {idx: item for item, idx in item2idx.items()}

    data = data.sort_values(by=["user"])
    data.reset_index(drop=True, inplace=True)

    idx_dict = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        "genre2idx": genre2idx,
        "writer2idx": writer2idx,
        "director2idx": director2idx,
        "year2idx": year2idx,
    }

    return data, field_dims, idx2user, idx2item, idx_dict


def inference_mapping(data, idx_dict, args):
    """
    최종 예측 값을 만들기 위한 데이터 프레임을 메타 데이터와 병합

    parameters
    ----------
    data : 예측할 batch_size 단위의 user와 모든 item
    idx_dict : 각 컬럼들을 mapping하기 위한 dict

    returns
    -------
    주어진 batch_size에 맞게 메타 데이터를 병합하고 tensor로 변환하여 반환
    """
    # 데이터 경로 선언
    path = args.dataset.data_path

    genre_data = os.path.join(path, "genres.tsv")
    writer_data = os.path.join(path, "writers.tsv")
    director_data = os.path.join(path, "directors.tsv")
    year_data = os.path.join(path, "years.tsv")
    title_data = os.path.join(path, "titles.tsv")

    # 데이터 불러오기
    genre_df = pd.read_csv(genre_data, sep="\t")
    genre_df = genre_df.drop_duplicates(subset=["item"])  # 중복 제거
    writer_df = pd.read_csv(writer_data, sep="\t")
    writer_df = writer_df.drop_duplicates(subset=["item"])  # 중복 제거
    director_df = pd.read_csv(director_data, sep="\t")
    director_df = director_df.drop_duplicates(subset=["item"])  # 중복 제거
    year_df = pd.read_csv(year_data, sep="\t")
    title_df = pd.read_csv(title_data, sep="\t")

    merged_data = data.copy()

    dfs = [genre_df, writer_df, director_df, year_df, title_df]

    for df in dfs:
        merged_data = pd.merge(merged_data, df, on="item", how="left").fillna(0)

    merged_data = handle_missing_value(merged_data)

    merged_data = merged_data.drop(columns=["title"])

    # user는 이미 임베딩되어 있는 상태
    merged_data["item"] = merged_data["item"].map(idx_dict["item2idx"])
    merged_data["genre"] = merged_data["genre"].map(idx_dict["genre2idx"])
    merged_data["writer"] = (
        merged_data["writer"].astype("string").map(idx_dict["writer2idx"])
    )
    merged_data["director"] = (
        merged_data["director"].astype("string").map(idx_dict["director2idx"])
    )
    merged_data["year"] = merged_data["year"].map(idx_dict["year2idx"])

    return torch.tensor(merged_data.values).long().to(args.device)


def negative_sampling(data, items):
    """
    Parameters
    ----------
    data : 'user', 'item'의 interaction이 포함된 데이터 프레임
    items : 전체 데이터셋의 고유 아이템 리스트
    """
    user_group_dfs = list(data.groupby("user")["item"])

    print("-----negative sampling-----")
    user_neg_dfs = []
    for u, u_items in tqdm(user_group_dfs):
        u_items = set(u_items)

        # negative sampling 개수 설정
        threshold = 500
        if len(u_items) >= threshold:
            num_negative = int(len(u_items) * 0.4)
        else:
            num_negative = int(len(u_items) * 0.2)

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
        user_neg_dfs.append(i_user_neg_df)

    user_neg_dfs = pd.concat(user_neg_dfs, axis=0, sort=False)
    data = pd.concat([data, user_neg_dfs], axis=0, sort=False)

    return data


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
        title = str(title)
        match = re.search(r"\((\d{4})\)", title)
        if match:
            return float(match.group(1))  # 'year'는 float64 타입이므로 변환
        return None

    data["year"] = data["year"].fillna(data["title"].apply(extract_year_from_title))

    return data


def train_valid_test_split(dataset):
    """
    train, valid를 9:1 비율로 분할
    """
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size]
    )
    return train_dataset, valid_dataset


def make_inference_data(data):
    """
    최종 예측을 위한 데이터 프레임 생성

    parameters
    ----------
    data : batch_size 만큼의 user와 해당 user와 상호작용한 item 조합
    """
    user_group_dfs = list(data.groupby("user")["item"])
    items = set(data["item"].unique())
    df_dict = {"user": [], "item": []}
    for u, u_items in user_group_dfs:
        u_items = set(u_items)
        i_user_neg_item = list(items - u_items)  # user와 상호작용하지 않은 item 리스트

        df_dict["user"].extend([u] * len(i_user_neg_item))
        df_dict["item"].extend(i_user_neg_item)
    inference_df = pd.DataFrame(df_dict)

    return inference_df


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

    # 최종 데이터 프레임 csv가 존재할 경우 불러오고 아닐 경우 생성
    if os.path.exists(os.path.join(path, "result_df.csv")):
        result_df = pd.read_csv(os.path.join(path, "result_df.csv"))
    else:
        rating_df = pd.read_csv(os.path.join(path, "train_ratings.csv"))
        result_df = rating_df.copy()
        result_df["interaction"] = 1.0  # 상호작용 여부
        result_df.drop(["time"], axis=1, inplace=True)  # 사용하지 않을 컬럼 drop

        # negative sampling
        items = set(result_df["item"])  # 고유 아이템 리스트
        result_df = negative_sampling(result_df, items)

        # genre 메타 데이터 불러오기
        genres_df = pd.read_csv(os.path.join(path, "genres.tsv"), delimiter="\t")
        genres_df = genres_df.drop_duplicates(
            subset=["item"]
        )  # 장르는 item 당 하나만 남김

        # director 메타 데이터 불러오기
        directors_df = pd.read_csv(os.path.join(path, "directors.tsv"), delimiter="\t")
        directors_df = directors_df.drop_duplicates(
            subset=["item"]
        )  # 감독은 item 당 하나만 남김

        # writer 메타 데이터 불러오기
        writers_df = pd.read_csv(os.path.join(path, "writers.tsv"), delimiter="\t")
        writers_df = writers_df.drop_duplicates(
            subset=["item"]
        )  # 작가는 item 당 하나만 남김

        # year 메타 데이터 불러오기
        years_df = pd.read_csv(os.path.join(path, "years.tsv"), delimiter="\t")
        titles_df = pd.read_csv(os.path.join(path, "titles.tsv"), delimiter="\t")

        dfs = [genres_df, writers_df, directors_df, years_df, titles_df]

        for df in dfs:
            result_df = pd.merge(result_df, df, on="item", how="left").fillna(0)

        # 결측치 처리
        result_df = handle_missing_value(result_df)

        result_df = result_df.drop(columns=["title"])

        # 최종 생성된 데이터 프레임 csv 파일로 저장
        result_df.to_csv(os.path.join(path, "result_df.csv"), mode="w", index=False)

    # 모든 컬럼 매핑
    result_df, field_dims, idx2user, idx2item, idx_dict = mapping(result_df)

    data = {
        "result_df": result_df,
        "idx2user": idx2user,
        "idx2item": idx2item,
        "idx_dict": idx_dict,
        "field_dims": field_dims,
    }

    return data
