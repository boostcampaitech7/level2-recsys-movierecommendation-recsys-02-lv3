import argparse
import ast
import os
import sys
from omegaconf import OmegaConf
import pandas as pd
import EASE as model_module
from inference import ease_predict

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

    data = setting.model_modular(args, 'dataset', 'data_load')(args)
    data_loader = setting.model_modular(args, 'dataloader')
    total_dataset = data_loader(args, data, datatype='total').data.toarray()
    
    ##### model load
    print("-"*15 + f"init {args.model}" + "-"*15)
    model = getattr(model_module, args.model)(args.model_args[args.model]['lambda'])

    ##### running model(train & evaluate & save model)
    print("-"*15 + f"{args.model} TRAINING" + "-"*15)
    model.train(total_dataset)

    ##### save predict
    print("-"*15 + f"SAVE {args.model} PREDICT" + "-"*15)    
    predict, top_items = ease_predict(args, model, total_dataset, 10)
    
    
    # output & index 정보 저장
    setting.save_file(args, predict)
    setting.save_file(args, data['id2user'], '.pkl', 'user')
    setting.save_file(args, data['id2item'], '.pkl', 'item')

    result = pd.DataFrame(top_items, columns=['user', 'item'])
    result['user'] = result['user'].apply(lambda x : data['id2user'][x])
    result['item'] = result['item'].apply(lambda x : data['id2item'][x])
    result = result.sort_values(by='user')

    filename = setting.get_submit_filename(args)
    result.to_csv(filename, index=False)

    print('Done!')





if __name__ == "__main__":



    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}
    
    
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', default='./choi/level2-recsys-movierecommendation-recsys-02-lv3/src/EASE/config.yaml')
    arg('--model', '-m', '--m', type=str, 
        default='EASE',
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
    
        
    config_yaml.model_args = OmegaConf.create({config_yaml.model : config_yaml.model_args[config_yaml.model]})


    print(OmegaConf.to_yaml(config_yaml))

    
    main(config_yaml)