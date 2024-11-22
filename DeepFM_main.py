import argparse
import ast
from omegaconf import OmegaConf
import pandas as pd
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
import src.models as model_module
from src.utils.util import Logger, Setting
from src.trainers.inference import deepfm_predict
from src.trainers.DeepFM_train import train, evaluate, test
from src.data.DeepFM.DeepFM_dataset import train_valid_test_split
import torch
import os


def main(args):

    ##### Setting
    Setting.seed_everything(args.seed)
    setting = Setting()

    if args.predict == False:
        log_path = setting.get_log_path(args)
        logger = Logger(args, log_path)
        logger.save_args()

    ##### data load
    print("-" * 15 + f"{args.model} Load Data" + "-" * 15)

    data = setting.model_modular(args, "dataset", "data_load")(args)

    data_X = data["result_df"].drop("interaction", axis=1)
    data_y = data["result_df"]["interaction"]

    X = torch.tensor(data_X.values).to(args.device)
    y = torch.tensor(data_y.values).to(args.device)

    data_loader = setting.model_modular(args, "dataloader")

    dataset = data_loader(args, X, y)
    train_dataset, valid_dataset, test_dataset = train_valid_test_split(dataset)

    ##### model load
    print("-" * 15 + f"init {args.model}" + "-" * 15)
    model = getattr(model_module, args.model)(args, data["field_dims"]).to(args.device)

    ##### running model(train & evaluate & save model)
    print("-" * 15 + f"{args.model} TRAINING" + "-" * 15)
    best_model = train(args, model, train_dataset, valid_dataset, logger, setting)

    ##### inference
    print("-" * 15 + f"{args.model} PREDICT" + "-" * 15)
    model = test(args, best_model, test_dataset, setting)

    ##### save predict
    print("-" * 15 + f"SAVE {args.model} PREDICT" + "-" * 15)

    total_dataset = pd.read_csv(
        os.path.join(args.dataset.data_path, "train_ratings.csv")
    )
    predicts = deepfm_predict(args, best_model, total_dataset, data["idx_dict"])

    result = pd.DataFrame(predicts, columns=["user", "item"])
    result["user"] = result["user"].apply(lambda x: data["idx2user"][x])
    result["item"] = result["item"].apply(lambda x: data["idx2item"][x])
    result = result.sort_values(by="user")

    filename = setting.get_submit_filename(args)
    result.to_csv(filename, index=False)

    print("Done!")


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
        required=True,
    )
    arg(
        "--predict",
        "-p",
        "--p",
        "--pred",
        type=ast.literal_eval,
        help="학습을 생략할지 여부를 설정할 수 있습니다.",
    )
    arg(
        "--model",
        "-m",
        "--m",
        type=str,
        choices=["MultiVAE", "DeepFM"],
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
    arg("--other_params", "--op", "-op", type=ast.literal_eval)
    arg("--optimizer", "-opt", "--opt", type=ast.literal_eval)
    arg("--lr_scheduler", "-lr", "--lr", type=ast.literal_eval)
    arg("--train", "-t", "--t", type=ast.literal_eval)

    args = parser.parse_args()

    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    # 사용되지 않는 정보 삭제 (학습 시에만)
    if config_yaml.predict == False:
        del config_yaml.checkpoint

        if config_yaml.wandb == False:
            del config_yaml.wandb_project, config_yaml.run_name

        config_yaml.model_args = OmegaConf.create(
            {config_yaml.model: config_yaml.model_args[config_yaml.model]}
        )

        config_yaml.optimizer.args = {
            k: v
            for k, v in config_yaml.optimizer.args.items()
            if k
            in getattr(
                optimizer_module, config_yaml.optimizer.type
            ).__init__.__code__.co_varnames
        }

        if config_yaml.lr_scheduler.use == False:
            del config_yaml.lr_scheduler.type, config_yaml.lr_scheduler.args
        else:
            config_yaml.lr_scheduler.args = {
                k: v
                for k, v in config_yaml.lr_scheduler.args.items()
                if k
                in getattr(
                    scheduler_module, config_yaml.lr_scheduler.type
                ).__init__.__code__.co_varnames
            }

        if config_yaml.train.resume == False:
            del config_yaml.train.resume_path

    # Configuration 콘솔에 출력
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
        )
        config_yaml.run_href = wandb.run.get_url()

        wandb.run.log_code(
            "./src"
        )  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능

    main(config_yaml)

    if config_yaml.wandb:
        wandb.finish()
