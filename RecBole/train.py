import argparse
import importlib
import os
from datetime import datetime

from recbole.config import Config
from recbole.trainer import Trainer
from recbole.data import create_dataset, data_preparation

from dataloader import generate_data

def get_data(config):
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config=config, dataset=dataset)
    return train_data, valid_data, test_data

def get_time():
    now = datetime.now()
    # 월, 일, 시간 추출
    month = now.strftime("%m")  # 월 (2자리 숫자)
    day = now.strftime("%d")    # 일 (2자리 숫자)
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    
    return f"{month}-{day}_{hour}-{minute}"

def train(config, train_data):
    model = model_class(config, tr_data.dataset).to(config['device'])
    trainer = Trainer(config,model)
    print("==== train start ====")
    best_valid_score, best_valid_result = trainer.fit(tr_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # model의 종류를 정의합니다 (general, sequential, context-aware)
    arg( "--type",
        "-t",
        type=str,
        choices=['g','s','c'],
        help="type of recommender model(g:general, s:sequential, c:context-aware)"
    )
    arg( "--model",
        "-m",
        type=str,
        help="name of recommender"
    )
    arg("--cur",
        default=get_time(),
        help="current date and time as string")
    args = parser.parse_args()
    
    generate_data(args=args)

    model_type = {'g':'general_recommender','s':'sequential_recommender','c':'context_aware_recommender'}
    recbole_model = importlib.import_module('recbole.model.'+model_type.get(args.type))
    model_class = getattr(recbole_model, args.model)

    config = Config(model=args.model, config_file_list=['Config/run.yaml', 'Config/setting.yaml'])
    config['checkpoint_dir'] = os.path.join(
        config['checkpoint_dir']+args.model+args.cur
    )

    tr_data, val_data, te_data = get_data(config)
    train(config=config, train_data=tr_data)