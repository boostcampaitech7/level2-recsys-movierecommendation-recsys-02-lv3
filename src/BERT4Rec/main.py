import os
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from BERT4Rec import BERT4Rec
from dataloader import get_dataloader
from train import train, evaluate
from inference import bert4rec_predict
from dataset import preprocess_data, preprocess_all_data

# from utils.util import Logger, Setting, transform_df_to_dict, get_total_probability


def main(config):
    """
    BERT4Rec을 사용한 모델 학습, 평가 및 추천 생성
        - 모델 및 학습 설정
        - 데이터 전처리
        - 모델 초기화 및 학습
        - 학습된 모델 평가
        - 학습된 모델 저장 및 불러오기
        - 추천 생성 및 제출 파일로 저장
    """

    # ##### Setting
    # Setting.seed_everything(args.seed)
    # setting = Setting()

    # Device 설정
    device = config.device if torch.cuda.is_available() else "cpu"

    # 데이터 전처리
    data_path = os.path.join(config.dataset.data_path, "train_ratings.csv")
    user_train, user_valid, num_user, num_item = preprocess_data(
        data_path, config.dataloader.batch_size
    )

    # Dataloader 생성
    dataloader = get_dataloader(
        user_train=user_train,
        num_user=num_user,
        num_item=num_item,
        max_len=config.model_args.BERT4Rec.max_len,
        mask_prob=config.other_params.args.mask_prob,
        batch_size=config.dataloader.batch_size,
    )

    # 모델 초기화
    torch.cuda.empty_cache()
    model = BERT4Rec(
        num_user=num_user,
        num_item=num_item,
        hidden_units=config.model_args.BERT4Rec.hidden_units,
        num_heads=config.model_args.BERT4Rec.num_heads,
        num_layers=config.model_args.BERT4Rec.num_layers,
        max_len=config.model_args.BERT4Rec.max_len,
        dropout_rate=config.model_args.BERT4Rec.dropout_rate,
        device=device,
    )
    model.to(device)

    # 손실 함수와 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = getattr(optim, config.optimizer.type)(
        model.parameters(), **config.optimizer.args
    )

    # 학습
    train(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        config.train.epochs,
        config.train.ckpt_dir,
    )

    # 최적 모델 로드
    best_model_path = os.path.join(config.train.ckpt_dir, "BERT4Rec_best.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path} for evaluation.")
    else:
        print("No best model found, using current model state.")

    # 평가
    evaluate(
        model,
        user_train,
        user_valid,
        num_user,
        num_item,
        config.model_args.BERT4Rec.max_len,
        device=device,
    )

    # # 모델 저장
    # os.makedirs(config.train.ckpt_dir, exist_ok=True)
    # model_path = os.path.join(config.train.ckpt_dir, "BERT4Rec.pt")
    # torch.save(model.state_dict(), model_path)

    # 저장된 모델 불러오기 및 추천 생성
    if config.predict:
        model.load_state_dict(torch.load(config.checkpoint))
        model.eval()

        user_train, num_user, num_item, idx2user, idx2item = preprocess_all_data(
            data_path
        )
        users = list(user_train.keys())
        output_path = os.path.join(config.train.submit_dir, "submission.csv")
        predict_save_path = os.path.join(config.train.submit_dir, "predictions.npy")
        os.makedirs(config.train.submit_dir, exist_ok=True)
        bert4rec_predict(
            model,
            users,
            user_train,
            num_user,
            num_item,
            config.model_args.BERT4Rec.max_len,
            idx2user,
            idx2item,
            output_path,
            predict_save_path,
            device,
        )


if __name__ == "__main__":
    # config.yaml 파일 로드
    config = OmegaConf.load("src/BERT4Rec/config.yaml")
    main(config)
