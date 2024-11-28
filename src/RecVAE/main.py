import argparse
import ast
import os
import sys
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.optim as optimizer_module
import RecVAE as model_module
from inference import recvae_predict
from dataloader import RecVAEDataset
from train import train, test

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.util import Logger, Setting


def main(args):

    
    ##### Setting
    Setting.seed_everything(args.seed)
    setting = Setting()
    
    
    if args.predict == False:
        log_path = setting.get_log_path(args)
        logger = Logger(args, log_path)
        logger.save_args()
    
    
    ##### data load    
    print("-"*15 + f"{args.model} Load Data" + "-"*15)

    data = setting.model_modular(args, 'dataset', 'data_load')(args)

    train_dataset = RecVAEDataset(args=args, data=data, datatype='train').data
    valid_dataset = RecVAEDataset(args=args, data=data, datatype='validation')
    test_dataset = RecVAEDataset(args=args, data=data, datatype='test')
    
    ##### model load
    print("-"*15 + f"init {args.model}" + "-"*15)
    model = getattr(model_module, args.model)(args.model_args[args.model]['hidden_dim'], args.model_args[args.model]['latent_dim'], args.model_args[args.model]['input_dim']).to(args.device)

    if args.predict == False:
        ##### running model(train & evaluate & save model)
        print("-"*15 + f"{args.model} TRAINING" + "-"*15)
        model = train(args, model, train_dataset, valid_dataset, logger, setting)

    ##### inference
    print("-"*15 + f"{args.model} PREDICT" + "-"*15)
    model = test(args, model, test_dataset, setting)


    ##### save predict
    print("-"*15 + f"SAVE {args.model} PREDICT" + "-"*15)
    
    total_dataset = RecVAEDataset(args=args, data=data, datatype='total').data
    
    # predict: ensemble 시 전체 아이템에 대한 확률값 필요시 사용
    predict, top_items = recvae_predict(args, model, total_dataset, k=20)
    
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
        help='Configuration 파일을 설정합니다.', default='/data/ephemeral/home/src/RecVAE/config.yaml')
    arg('--predict', '-p', '--p', '--pred', type=ast.literal_eval, 
        help='학습을 생략할지 여부를 설정할 수 있습니다.')
    arg('--model', '-m', '--m', type=str, 
        default='RecVAE',
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--wandb', '--w', '-w', type=ast.literal_eval, 
        help='wandb를 사용할지 여부를 설정할 수 있습니다.', default= False)
    arg('--wandb_project', '--wp', '-wp', type=str,
        help='wandb 프로젝트 이름을 설정할 수 있습니다.')
    arg('--run_name', '--rn', '-rn', '--r', '-r', type=str,
        help='wandb에서 사용할 run 이름을 설정할 수 있습니다.')
    arg('--model_args', '--ma', '-ma', type=ast.literal_eval)
    arg('--dataloader', '--dl', '-dl', type=ast.literal_eval)
    arg('--dataset', '--dset', '-dset', type=ast.literal_eval)
    arg('--other_params', '--op', '-op', type=ast.literal_eval)
    arg('--optimizer', '-opt', '--opt', type=ast.literal_eval)
    arg('--lr_scheduler', '-lr', '--lr', type=ast.literal_eval)
    arg('--dropout', type=int, default=0.5)
    arg('--step', type=int, default=10)
    arg('--gamma', type=float, default=0.004)
    arg('--lambd', type=float, default=500)
    arg('--alpha', type=float, default=1)
    arg('--threshold', type=int, default=1000)
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
    
        if config_yaml.wandb == False:
            del config_yaml.wandb_project, config_yaml.run_name
        
        config_yaml.model_args = OmegaConf.create({config_yaml.model : config_yaml.model_args[config_yaml.model]})
        
        config_yaml.optimizer.args = {k: v for k, v in config_yaml.optimizer.args.items() 
                                    if k in getattr(optimizer_module, config_yaml.optimizer.type).__init__.__code__.co_varnames}

        if config_yaml.train.resume == False:
            del config_yaml.train.resume_path

    # Configuration 콘솔에 출력
    print(OmegaConf.to_yaml(config_yaml))
    
    
    ######################## W&B
    if config_yaml.wandb:
        import wandb
        wandb.init(project=config_yaml.wandb_project, 
                   config=OmegaConf.to_container(config_yaml, resolve=True),
                   name=config_yaml.run_name if config_yaml.run_name else None,
                   notes=config_yaml.memo if hasattr(config_yaml, 'memo') else None,
                   tags=[config_yaml.model],
                   resume="allow")
        config_yaml.run_href = wandb.run.get_url()

        wandb.run.log_code("./src/RecVAE")
    
    
    main(config_yaml)
    
    if config_yaml.wandb:
        wandb.finish()