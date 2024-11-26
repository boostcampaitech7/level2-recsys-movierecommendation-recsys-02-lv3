import subprocess
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from src.utils.util import Logger, Setting
from src.utils.util import optimize_replace_inf, row_min_max_normalization


def main(args):

    ##### Setting
    Setting.seed_everything(args.seed)
    setting = Setting()
    
    model_list = args.models
    weight_list = args.ensemble_weights
    
    ensemble_predict = np.zeros((31360, 6807))
    weighted_sum = 0
    
    for model, weight in tqdm(zip(model_list, weight_list), total=len(model_list), desc="Processing models"):
        print('-'*10, f'{model} Load', '-'*10)
        predict = setting.get_latest_file(args.predict_dir, model)
        arrays = np.load(predict)
        
        user_matrix, item_matrix = setting.get_latest_file(args.predict_dir, model, file_extension='.pkl')
        
        user_dict = dict(sorted(user_matrix.items(), key=lambda item: item[1]))
        item_dict = dict(sorted(item_matrix.items(), key=lambda item: item[1]))

        if model == 'EASE':
            k = 0.1
        elif model == 'MultiVAE':
            k = 1
            
        print('-'*10, f'{model} Preprocessing', '-'*10)
        replace_predict = optimize_replace_inf(torch.tensor(arrays, dtype=torch.float64), k)    # -inf 값 처리
        scaled_predict = row_min_max_normalization(replace_predict)
        sorted_predict = scaled_predict[torch.tensor(np.array(list(user_dict.keys()))),:][:, torch.tensor(np.array(list(item_dict.keys())))]
        
        print('-'*10, f'{model} data ensemble', '-'*10)
        ensemble_predict += float(weight) * sorted_predict.cpu().numpy()
        weighted_sum += float(weight)
        
    new_user = {index: value for index, (key, value) in enumerate(user_dict.items())}
    new_item = {index: value for index, (key, value) in enumerate(item_dict.items())}    
        
    ensemble_output = ensemble_predict / weighted_sum

    print('-'*10, f'Extract Top-K Item', '-'*10)
    top_items = []

    _  , top_ids = torch.topk(torch.from_numpy(ensemble_output).to(args.device), 10, dim=1)
    for user_id, item_ids in enumerate(top_ids):
        for item_id in item_ids:
            top_items.append((user_id, item_id.item()))

    print('-'*10, f'Submission', '-'*10)
    result = pd.DataFrame(top_items, columns=['user', 'item'])
    result['user'] = result['user'].apply(lambda x : new_user[x])
    result['item'] = result['item'].apply(lambda x : new_item[x])
    result = result.sort_values(by='user')


    print('-'*10, f'Save', '-'*10)
    file_path = setting.make_dir(args.submit_dir)
    models_combined = ",".join(model_list)
    filename = os.path.join(file_path, f"ensemble({models_combined}).csv")
    result.to_csv(filename, index=False)

    print('Done!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}
    
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', default='choi/level2-recsys-movierecommendation-recsys-02-lv3/configs/config.yaml')    
    arg('--models', '-m', '--m', nargs="+", required=True,
        help='앙상블에 사용할 모델 list를 설정합니다.')
    arg('--seed', '-s', '--s', type=int, default=0,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--scaling_method', '-sm', type=str, choices=['min_max', 'softmax', 'average'], required=True, 
        help='scaling에 사용할 방법 선택합니다. (min_max, softmax, average 등)')
    arg('--ensemble_weights', '-ew', nargs="+", required=True, 
        help='ensemble에 사용할 각 모델별 weight list를 반환합니다.(model과 동일한 순서로 설정)')
    
    
    args = parser.parse_args()
    
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    del config_yaml.model_names
    
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    print(OmegaConf.to_yaml(config_yaml))

    main(config_yaml)

    