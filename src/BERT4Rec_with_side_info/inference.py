from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

def get_genre_seq(seq, item_genre_dic):
    lst = []
    for i in seq:
        selected_genre_list = item_genre_dic[i]
        padding_len = 3 - len(selected_genre_list)
        if len(selected_genre_list) >= 3:
            lst.append(selected_genre_list[:3])
        else:
            padded_genre_list = [0] * padding_len + selected_genre_list
            lst.append(padded_genre_list)
    return lst


def inference(model, num_user, num_item, train_df, user2idx, item2idx, max_len, item_genre_dic):
    # inference
    model.eval()
    predict_list = []
    for u in tqdm(range(num_user)):
        seq = (train_df[u] + [num_item + 1])[-max_len :]
        genre_seq = get_genre_seq(seq, item_genre_dic)

        used_items_list = [
            a - 1 for a in train_df[u]
        ]  # 사용한 아이템에 대해 인덱스 계산을 위해 1씩 뺀다.

        if len(seq) < max_len:
            genre_seq = np.array([[0, 0, 0] for _ in range(max_len - len(seq))] + genre_seq)

            seq = np.pad( 
                seq,
                (max_len - len(seq), 0),
                "constant",
                constant_values=0,
            )  # 패딩 추가


        with torch.no_grad():
            predictions = -model(np.array([seq]), np.array(genre_seq))
            predictions = predictions[0][-1][1:]  # mask 제외
            predictions[used_items_list] = np.inf  # 사용한 아이템은 제외하기 위해 inf
            rank = predictions.argsort().argsort().tolist()

            for i in range(10):
                rank.index(i)
                predict_list.append([u, rank.index(i)])

    # Data Export
    # 인덱스를 원래 데이터 상태로 변환하여 csv 저장합니다.
    predict_list_idx = [
        [user2idx.index[user], item2idx.index[item]] for user, item in predict_list
    ]
    predict_df = pd.DataFrame(data=predict_list_idx, columns=["user", "item"])
    predict_df = predict_df.sort_values("user")

    return predict_df