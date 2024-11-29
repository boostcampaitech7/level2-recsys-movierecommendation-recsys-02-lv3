import yaml
import argparse
import ast
import os
import sys
from omegaconf import OmegaConf
from datetime import datetime
from pytz import timezone
import sys
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

from dataloader import *
from dataset import MakeMatrixDataSet
import EASER as model_module

import warnings
warnings.filterwarnings(action="ignore")

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.util import Setting


def main(args):
    ##### Setting
    Setting.seed_everything(args.seed)
    setting = Setting()
    
    
    ##### data load    
    print("-"*15 + f"{args.model} Load Data" + "-"*15)
    data_loader = Dataloader(args)
    train_df, users, items = data_loader.dataloader()
    
    print("-"*15 + f"init {args.model}" + "-"*15)
    model = getattr(model_module, args.model)(args)
   
   
    print("-"*15 + f"SAVE {args.model} PREDICT" + "-"*15)    
    predict = model.fit(train_df)
    setting.save_file(args, predict)
    
    # 결과 저장 (유저별 Top-K 추천)
    result_df, user_index_to_id, item_index_to_id = model.predict(train_df, users, items, args.train.K)
    setting.save_file(args, user_index_to_id, '.pkl', 'user')
    setting.save_file(args, item_index_to_id, '.pkl', 'item')
    
    result_df = result_df[['user','item']].sort_values(by='user')

    filename = setting.get_submit_filename(args)
    result_df.to_csv(filename, index=False)

    print('Done!')


   
if __name__ == "__main__":



    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}
    
    
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', default='src/EASER/config.yaml')
    arg('--model', '-m', '--m', type=str, 
        default='EASER',
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    arg('--train', '-t', '--t', type=ast.literal_eval)    
    
    args = parser.parse_args()
    
    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    print(OmegaConf.to_yaml(config_yaml))

    main(config_yaml)