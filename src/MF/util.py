import numpy as np
from typing import Tuple
from tqdm import tqdm
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def evaluate_recall(answers, pred_list):
    recall = []
    for k in [5, 10]:
        recall.append(recall_at_k(answers, pred_list, k))
    post_fix = {
        "RECALL@5": "{:.4f}".format(recall[0]),
        "RECALL@10": "{:.4f}".format(recall[1]),
    }
    print(post_fix)

    return [recall[0], recall[1]], str(post_fix)


def ratings_answer_split(ratings, valid) -> Tuple[pd.DataFrame, list]:
    sub_users = []
    sub_items = []
    sub_answers = []

    user_positives = ratings.groupby("user")["item"].apply(list)
    for u, items in tqdm(
        user_positives.items(), total=len(user_positives), desc="get subset"
    ):
        lasts = items[-1:]
        cp_items = items[:-1]

        pops = []
        for i in range(valid):
            pops.append(cp_items.pop(np.random.randint(len(cp_items))))
        sub_answers.append(lasts + pops)
        for i in cp_items:
            sub_users.append(u)
            sub_items.append(i)

    sub_ratings = pd.DataFrame(
        {
            "user": sub_users,
            "item": sub_items,
        }
    )
    return sub_ratings, sub_answers


def generate_submission_file(data_file, preds, filename):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(filename, index=False)


def data_load(args):
    ratings = pd.read_csv(args.data_path + "train_ratings.csv")
    titles = pd.read_csv(args.data_path + "titles.tsv", sep="\t")
    genres = pd.read_csv(args.data_path + "genres.tsv", sep="\t")

    genres = genres.groupby("item")["genre"].agg(list).reset_index()
    genres_and_titles = pd.merge(titles, genres, on="item", how="left")
    genres_and_titles["text"] = (
        genres_and_titles["title"]
        + " "
        + genres_and_titles["genre"].apply(lambda x: " ".join(x))
    )

    user2idx = {v: k for k, v in enumerate(ratings["user"].unique())}
    idx2user = {k: v for k, v in enumerate(ratings["user"].unique())}
    item2idx = {v: k for k, v in enumerate(ratings["item"].unique())}
    idx2item = {k: v for k, v in enumerate(ratings["item"].unique())}
    ratings["user"] = ratings["user"].map(user2idx)
    ratings["item"] = ratings["item"].map(item2idx)
    genres_and_titles["item"] = genres_and_titles["item"].map(item2idx)
    genres_and_titles = genres_and_titles.sort_values(by="item", ascending=True)
    genres_and_titles.reset_index(drop=True, inplace=True)

    tfidf_matrix_path = args.model_args[args.model].data_path + "tfidf/tfidf_matrix.pkl"
    if os.path.exists(tfidf_matrix_path):
        # Load the previously saved TF-IDF matrix
        with open(tfidf_matrix_path, "rb") as f:
            tfidf_matrix = pickle.load(f)
    else:
        if not os.path.exists(args.model_args[args.model].data_path + "tfidf/"):
            os.mkdir(args.model_args[args.model].data_path + "tfidf/")
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=args.model_args[args.model].feature_dims
        )  # Adjust as needed
        tfidf_matrix = tfidf.fit_transform(genres_and_titles["text"])
        # Save the TF-IDF matrix for future use
        with open(tfidf_matrix_path, "wb") as f:
            pickle.dump(tfidf_matrix, f)

    data = {
        "ratings": ratings,
        "tfidf_matrix": tfidf_matrix,
        "user2idx": user2idx,
        "idx2user": idx2user,
        "item2idx": item2idx,
        "idx2item": idx2item,
    }

    return data
