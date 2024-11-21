import argparse
import importlib
import os
from omegaconf import OmegaConf
from datetime import datetime

from recbole.config import Config
from recbole.trainer import Trainer
import sys
sys.path.insert(0, "/data/ephemeral/home")
from src.utils.util import Logger, Setting
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

def train(config,logger):
    ### data initialize
    tr_data, val_data, te_data = get_data(config)
    
    ### model initialize
    model = model_class(config, tr_data.dataset).to(config['device'])
    trainer = Trainer(config,model)

    print("==== train start ====")
    best_valid_score, best_valid_result = trainer.fit(train_data=tr_data,valid_data=val_data)

    print("==== train end! ====")
    logger.info(f"best valid score: {best_valid_score}")
    logger.info(f"best valid result: {best_valid_result}")
    print(f"best valid score & result: {best_valid_score}, {best_valid_result}")
    
    print("==== evaluation start ====")
    test_result = trainer.evaluate(te_data)
    logger.info(f"test result: {test_result}")
    print(f"test_result: {test_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    # model의 종류를 정의합니다 (general, sequential, context-aware)
    arg("--config", 
        "-c",
        help='Configuration 파일을 설정합니다.',
        default='../configs/config.yaml'
    )
    arg("--root_dir",
        "-r",
        type=str,
        default='/data/ephemeral/home'
    )
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

    #### Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config)

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]
    print("config args", args)
    args = config_yaml

    ##### setting
    setting = Setting()

    ##### logger initialize
    logger = Logger(args, setting.get_log_path(args))
    logger.save_args()

    model_type = {'g':'general_recommender','s':'sequential_recommender','c':'context_aware_recommender'}
    recbole_model = importlib.import_module('recbole.model.'+model_type.get(args.type))
    model_class = getattr(recbole_model, args.model)

    config = Config(model=args.model, config_file_list=['../configs/recbole_model.yaml', '../configs/recbole_setting.yaml'])

    config['checkpoint_dir'] = os.path.join(
        args.root_dir,args.train.ckpt_dir,args.model+'_'+args.cur
    )
    print(config)    
    generate_data(args=args, config=config)
    train(config=config, logger=logger)