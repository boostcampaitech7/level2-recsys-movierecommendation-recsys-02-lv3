import sys
import os
import torch
import numpy as np
from tqdm import tqdm
from util import data_load, ratings_answer_split, generate_submission_file
from MF import als
from omegaconf import OmegaConf
import argparse
import ast

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.util import Setting


def main(args):

    ##### Setting
    Setting.seed_everything(args.seed)
    setting = Setting()

    ##### data load
    print("-" * 15 + f"{args.model} Load Data" + "-" * 15)

    data = data_load(args)

    args.model_args[args.model].n_users = data["ratings"]["user"].nunique()
    args.model_args[args.model].n_items = data["ratings"]["item"].nunique()

    ratings = data["ratings"]

    if not args.model_args[args.model].inference:  # 제출 파일 생성 x
        ratings, answers = ratings_answer_split(
            ratings, args.model_args[args.model].valid
        )

    positives = ratings.groupby("user")["item"].apply(list)

    if not os.path.exists(args.model_args[args.model].data_path + "MF_model/"):
        os.mkdir(args.model_args[args.model].data_path + "MF_model/")
    if not args.model_args[args.model].inference:
        rating_mat_path = (
            args.model_args[args.model].data_path
            + f"MF_model/rating_mat_v{args.model_args[args.model].valid}_seed{args.seed}.npy"
        )
        if not os.path.exists(rating_mat_path):
            rating_mat = torch.zeros(
                (
                    args.model_args[args.model].n_users,
                    args.model_args[args.model].n_items,
                ),
                dtype=torch.float32,
            )
            for u, items in tqdm(
                positives.items(), total=len(positives), desc=f"make valid rating mat"
            ):
                for i in items:
                    rating_mat[u][i] = 1.0
            np.save(rating_mat_path, rating_mat.numpy())
        else:
            rating_mat = np.load(rating_mat_path)
            rating_mat = torch.from_numpy(rating_mat)
    else:
        rating_mat = torch.zeros(
            (args.model_args[args.model].n_users, args.model_args[args.model].n_items),
            dtype=torch.float32,
        )
        for u, items in tqdm(
            positives.items(), total=len(positives), desc="make inference rating mat"
        ):
            for i in items:
                rating_mat[u][i] = 1.0

    ##### model load
    print("-" * 15 + f"init {args.model}" + "-" * 15)

    if not args.model_args[args.model].inference:
        rating_mat = torch.zeros(
            (args.model_args[args.model].n_users, args.model_args[args.model].n_items),
            dtype=torch.float32,
        )
        for u, items in tqdm(
            positives.items(), total=len(positives), desc=f"make valid rating mat"
        ):
            for i in items:
                rating_mat[u][i] = 1.0
    else:
        rating_mat = torch.zeros(
            (args.model_args[args.model].n_users, args.model_args[args.model].n_items),
            dtype=torch.float32,
        )
        for u, items in tqdm(
            positives.items(), total=len(positives), desc="make inference rating mat"
        ):
            for i in items:
                rating_mat[u][i] = 1.0

    rating_mat = rating_mat.to(args.device)
    preds = als(
        args,
        rating_mat,
        answers if not args.model_args[args.model].inference else None,
        alpha=args.model_args[args.model].alpha,
        epochs=args.model_args[args.model].epochs,
        l1=args.model_args[args.model].l1,
        feature_dim=args.model_args[args.model].feature_dims,
        inference=args.model_args[args.model].inference,
        tfidf_matrix=data["tfidf_matrix"],
        tfidf_weight=args.model_args[args.model].tfidf_weight,
    )

    # 제출 파일 생성
    if args.model_args[args.model].inference:
        if not os.path.exists(args.model_args[args.model].output_dir):
            os.mkdir(args.model_args[args.model].output_dir)

        item_preds = []
        print("Submission label encoding... ", flush=True)
        for pred in tqdm(preds):
            item_pred = [data["idx2item"][idx] for idx in pred]  # Reverse the encoding
            item_preds.append(item_pred)

        item_preds = np.array(item_preds)
        filename = setting.get_submit_filename(args)

        generate_submission_file(
            args.model_args[args.model].data_path + "train_ratings.csv",
            item_preds[:, :10],
            filename,
        )
        print("done.")


if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description="parser")

    arg = parser.add_argument
    str2dict = lambda x: {k: int(v) for k, v in (i.split(":") for i in x.split(","))}

    arg(
        "--config",
        "-c",
        "--c",
        type=str,
        help="Configuration 파일을 설정합니다.",
        default="./choi/level2-recsys-movierecommendation-recsys-02-lv3/src/EASE/config.yaml",
    )
    arg(
        "--model",
        "-m",
        "--m",
        type=str,
        default="MF",
        help="학습 및 예측할 모델을 선택할 수 있습니다.",
    )
    arg(
        "--seed",
        "-s",
        "--s",
        type=int,
        help="데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.",
    )
    arg(
        "--checkpoint",
        "-ckpt",
        "--ckpt",
        type=str,
        help="학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.",
    )
    arg(
        "--device",
        "-d",
        "--d",
        type=str,
        choices=["cuda", "cpu", "mps"],
        help="사용할 디바이스를 선택할 수 있습니다.",
    )
    arg(
        "--wandb",
        "--w",
        "-w",
        type=ast.literal_eval,
        help="wandb를 사용할지 여부를 설정할 수 있습니다.",
        default=False,
    )
    arg(
        "--wandb_project",
        "--wp",
        "-wp",
        type=str,
        help="wandb 프로젝트 이름을 설정할 수 있습니다.",
    )
    arg(
        "--run_name",
        "--rn",
        "-rn",
        "--r",
        "-r",
        type=str,
        help="wandb에서 사용할 run 이름을 설정할 수 있습니다.",
    )
    arg("--model_args", "--ma", "-ma", type=ast.literal_eval)
    arg("--dataloader", "--dl", "-dl", type=ast.literal_eval)
    arg("--dataset", "--dset", "-dset", type=ast.literal_eval)
    arg("--train", "-t", "--t", type=ast.literal_eval)
    arg("--topk", default=10, type=int)
    arg("--feature_dims", default=64, type=int)
    arg("--alpha", default=1, type=int)
    arg("--l1", default=0.5, type=int)
    arg("--data_path", default="../data/train/", type=str)
    arg("--output_dir", default="../saved/submit/", type=str)
    arg("--inference", default=False, action="store_true")

    args = parser.parse_args()

    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    if config_yaml.wandb == False:
        del config_yaml.wandb_project, config_yaml.run_name

    config_yaml.model_args = OmegaConf.create(
        {config_yaml.model: config_yaml.model_args[config_yaml.model]}
    )

    print(OmegaConf.to_yaml(config_yaml))

    ######################## W&B
    if config_yaml.wandb:
        import wandb

        wandb.init(
            project=config_yaml.wandb_project,
            config=OmegaConf.to_container(config_yaml, resolve=True),
            name=config_yaml.run_name if config_yaml.run_name else None,
            notes=config_yaml.memo if hasattr(config_yaml, "memo") else None,
            tags=[config_yaml.model],
            resume="allow",
            group=f"Sweep_{config_yaml.model}",
        )
        config_yaml.run_href = wandb.run.get_url()

        wandb.run.log_code("./src/MF")

    main(config_yaml)

    if config_yaml.wandb:
        wandb.finish()
