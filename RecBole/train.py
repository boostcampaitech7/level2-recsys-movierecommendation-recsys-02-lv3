import argparse
import importlib
import os
from datetime import datetime

from recbole.config import Config
from recbole.trainer import Trainer

from dataloader import generate_data, get_data

# checkpoint 생성 시 모델 학습 시각 기록을 위한 함수
def get_time():
    now = datetime.now()
    # 월, 일, 시간 추출
    month = now.strftime("%m")  # 월 (2자리 숫자)
    day = now.strftime("%d")    # 일 (2자리 숫자)
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    
    return f"{month}-{day}_{hour}-{minute}"

# train 함수에서 Logger, Wandb dev 브랜치 버전으로 업데이트 예정
def train(config):
    tr_data, val_data, te_data = get_data(config)
    model = model_class(config, tr_data.dataset).to(config['device'])
    trainer = Trainer(config,model)
    print("==== train start ====")
    best_valid_score, best_valid_result = trainer.fit(train_data=tr_data,valid_data=val_data)
    print("==== train end! ====")
    print(f"best valid score & result: {best_valid_score}, {best_valid_result}")
    print("==== evaluation start ====")
    test_result = trainer.evaluate(te_data)
    print(f"test_result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # model의 종류를 정의합니다 (general, sequential, context-aware)
    arg("--type",
        "-t",
        type=str,
        choices=['g','s','c'],
        help="type of recommender model(g:general, s:sequential, c:context-aware)"
    )
    arg("--model",
        "-m",
        type=str,
        help="name of recommender"
    )
    arg("--data_path",
        "-d",
        type=str,
        help="path of saved data",
        default='./data/'
    )
    arg("--cur",
        default=get_time(),
        help="current date and time as string")
    args = parser.parse_args()

    model_type = {'g':'general_recommender','s':'sequential_recommender','c':'context_aware_recommender'}
    recbole_model = importlib.import_module('recbole.model.'+model_type.get(args.type))
    model_class = getattr(recbole_model, args.model)

    config = Config(model=args.model, config_file_list=['Config/run.yaml', 'Config/setting.yaml'])
    config['checkpoint_dir'] = os.path.join(
        config['checkpoint_dir']+args.model+args.cur
    )

    generate_data(args=args, config=config)
    train(config=config)