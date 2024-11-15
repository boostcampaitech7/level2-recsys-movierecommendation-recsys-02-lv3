import argparse
import ast
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from src.trainers import train, inference
import src.models as model_module
from src.data.dataloader import MultiVAE_DataLoader
from src.data.dataset import data_load
import warnings
from src.utils.util import Setting
import os

def main(args):

    
    ##### Setting
    Setting.seed_everything(args.seed)
    setting = Setting()
    filename = setting.get_submit_filename(args)
    
    
    ##### data load    
    print("-"*15 + f"{args.model} Load Data" + "-"*15)
    
    data = data_load(args)
    loader = MultiVAE_DataLoader(args, data)
    n_items = len(data['unique_sid'])
    p_dims = args.model_args[args.model]['p_dims'] + [n_items]


    ##### model load
    print("-"*15 + f"init {args.model}" + "-"*15)
    model = getattr(model_module, args.model)(p_dims).to(args.device)

    ##### running model(train & evaluate & save model)
    print("-"*15 + f"{args.model} TRAINING" + "-"*15)
    best_model = train.run(args, model, loader, setting)

    ##### inference
    print(f'--------------- {args.model} PREDICT ---------------')
    train_data = loader.load_data('train', True)
    predicts = inference.predict(args, model, train_data)

    ##### save predict
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    result = pd.DataFrame(predicts, columns=['user', 'item'])
    result['user'] = result['user'].apply(lambda x : data['id2profile'][x])
    result['item'] = result['item'].apply(lambda x : data['id2show'][x])
    result = result.sort_values(by='user')

    write_path = os.path.join(filename)
    result.to_csv(write_path, index=False)



if __name__ == "__main__":

    ## 각종 파라미터 세팅
    
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')

    arg = parser.add_argument

    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', required=True)
    arg('--model', '-m', '--m', type=str, 
        choices=['MultiVAE'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--run_name', '--rn', '-rn', '--r', '-r', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    arg('--other_params', '--op', '-op', type=ast.literal_eval)
    arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    arg('--lr_scheduler', '-lr', '--lr', type=ast.literal_eval)
    arg('--train', '-t', '--t', type=ast.literal_eval)
    
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
    
        config_yaml.model_args = OmegaConf.create({config_yaml.model : config_yaml.model_args[config_yaml.model]})
        
        config_yaml.optimizer.args = {k: v for k, v in config_yaml.optimizer.args.items() 
                                    if k in getattr(optimizer_module, config_yaml.optimizer.type).__init__.__code__.co_varnames}
        
        if config_yaml.lr_scheduler.use == False:
            del config_yaml.lr_scheduler.type, config_yaml.lr_scheduler.args
        else:
            config_yaml.lr_scheduler.args = {k: v for k, v in config_yaml.lr_scheduler.args.items() 
                                            if k in getattr(scheduler_module, config_yaml.lr_scheduler.type).__init__.__code__.co_varnames}
        
        if config_yaml.train.resume == False:
            del config_yaml.train.resume_path

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))
    
    main(config_yaml)