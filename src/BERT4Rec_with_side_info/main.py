import argparse
import ast
from omegaconf import OmegaConf
import pandas as pd
import os
import sys
import glob

import torch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from torch.utils.data import DataLoader

from inference import inference
from train import train
from metrics import recall_at_k

current_dir = os.path.dirname(os.path.abspath(__file__)) 
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.util import Logger, Setting, transform_df_to_dict, get_total_probability

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

    data_loader = setting.model_modular(args, 'dataloader')

    train_dataset = data_loader(data['train_df'], data['num_user'], data['num_item'], args.model_args.BERT4Rec_with_side_info.hidden_units, 
                                args.model_args.BERT4Rec_with_side_info.max_len, args.train.mask_prob, data['item_genre_dic'])
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.train.batch_size, pin_memory=True
    )

    valid_dataset = data_loader(data['valid_df'], data['num_user'], data['num_item'], args.model_args.BERT4Rec_with_side_info.hidden_units, 
                                args.model_args.BERT4Rec_with_side_info.max_len, args.train.mask_prob, data['item_genre_dic'])
    valid_dataloader = DataLoader(
        valid_dataset, shuffle=False, batch_size=args.train.batch_size, pin_memory=True
    )

    
    ##### model load
    print("-"*15 + f"init {args.model}" + "-"*15)
    model = setting.model_modular(args, 'model', 'BERT4Rec')(data['num_user'], data['num_item'], data['num_genres'],
                                              args.model_args.BERT4Rec_with_side_info.hidden_units, args.model_args.BERT4Rec_with_side_info.num_heads, args.model_args.BERT4Rec_with_side_info.num_layers, 
                                              args.model_args.BERT4Rec_with_side_info.max_len, args.model_args.BERT4Rec_with_side_info.dropout_rate, args.device)

    model = model.to(args.device)
    
    if os.path.exists(args.train.ckpt_dir):
        print("Load model from pretrained folder")
        model_path = os.path.join(args.train.ckpt_dir, 'best.pt')
        model_state_dict = torch.load(model_path, map_location=args.device)
        model.load_state_dict(model_state_dict)

    else:
        ##### running model(train & evaluate & save model)
        print("-"*15 + f"{args.model} TRAINING" + "-"*15)
        model = train(args, model, train_dataloader, valid_dataloader, logger)

    ##### inference
    print("-"*15 + f"{args.model} INFERENCE" + "-"*15)
    predict_df, _ = inference(model, data['num_user'], data['num_item'], data['train_df'], data['user2idx'], data['item2idx'], args.model_args.BERT4Rec_with_side_info.max_len, data['item_genre_dic'])

    transform_predict_df = predict_df.copy()
    transform_predict_df['user_idx'] = transform_predict_df['user'].apply(lambda x: data['user2idx'][x])
    transform_predict_df['item_idx'] = transform_predict_df['item'].apply(lambda x: data['item2idx'][x])

    transform_predict_df = transform_df_to_dict(transform_predict_df)
    test_data = transform_df_to_dict(data['test_data'])
    
    ##### evaluate
    recall_at_k(data['num_user'], transform_predict_df, test_data)

    ##### predict with total data
    print("-"*15 + f"SAVE {args.model} PREDICT" + "-"*15)
    final_predict, logit_list = inference(model, data['num_user'], data['num_item'], data['total_df'], data['user2idx'], data['item2idx'], args.model_args.BERT4Rec_with_side_info.max_len, data['item_genre_dic'])
    
    # save the submit & ensemble file
    sorted_probabilities = get_total_probability(logit_list)

    setting.save_file(args, sorted_probabilities, file_extension=".npy", type = None)
    setting.save_file(args, data['idx2user'], '.pkl', 'user')
    setting.save_file(args, data['idx2item'], '.pkl', 'item')

    filename = setting.get_submit_filename(args)
    final_predict.to_csv(filename, index=False)

    print('Done!')



if __name__ == "__main__":



    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}
    

    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', default = '/data/ephemeral/home/level2-recsys-movierecommendation-recsys-02-lv3/src/BERT4Rec_with_side_info/config.yaml')
    arg('--predict', '-p', '--p', '--pred', type=ast.literal_eval, 
        help='학습을 생략할지 여부를 설정할 수 있습니다.')
    arg('--model', '-m', '--m', type=str, 
        choices=['MultiVAE'],
        help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--seed', '-s', '--s', type=int,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--checkpoint', '-ckpt', '--ckpt', type=str, 
        help='학습을 생략할 때 사용할 모델을 설정할 수 있습니다. 단, 하이퍼파라미터 세팅을 모두 정확하게 입력해야 합니다.')
    arg('--device', '-d', '--d', type=str, 
        choices=['cuda', 'cpu', 'mps'], help='사용할 디바이스를 선택할 수 있습니다.')
    arg('--wandb', '--w', '-w', type=ast.literal_eval, 
        help='wandb를 사용할지 여부를 설정할 수 있습니다.')
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
        
        if config_yaml.lr_scheduler.use == False:
            del config_yaml.lr_scheduler.type, config_yaml.lr_scheduler.args
        else:
            config_yaml.lr_scheduler.args = {k: v for k, v in config_yaml.lr_scheduler.args.items() 
                                            if k in getattr(scheduler_module, config_yaml.lr_scheduler.type).__init__.__code__.co_varnames}
        
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

        wandb.run.log_code("./src")  # src 내의 모든 파일을 업로드. Artifacts에서 확인 가능
    
    
    main(config_yaml)
    
    if config_yaml.wandb:
        wandb.finish()