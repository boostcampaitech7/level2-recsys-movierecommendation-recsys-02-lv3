import argparse
import os
from omegaconf import OmegaConf

from recbole.config import Config
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from dataloader import get_data

import torch
import numpy as np
import pandas as pd


def inference(args, config):
    _, _, te_data = get_data(config)

    (
        inference_config,
        model,
        inference_dataset,
        _,
        _,
        _,
    ) = load_data_and_model(
        os.path.join(
            args.root_dir,args.train.ckpt_dir,args.ckpt_file
    ))
    
    user_id2token = inference_dataset.field2id_token['user_id']
    item_id2token = inference_dataset.field2id_token['item_id']

    all_user_list = torch.arange(1, len(user_id2token)).view(
        -1, len(user_id2token)-1
    )

    pred_list = None

    model.eval()
    model.to(inference_config['device'])
    for user in all_user_list:
        batch_pred_scores, batch_pred_list = full_sort_topk(
            user,
            model,
            te_data,
            10,
            device='cuda',
        )

    batch_pred_list = batch_pred_list.clone().detach().cpu().numpy()
    batch_pred_scores = batch_pred_scores.clone().detach().cpu().numpy()

    
    if pred_list is None:
        pred_list = batch_pred_list
        pred_scores = batch_pred_scores
        user_list = user.numpy()
    else:
        pred_list = np.append(pred_list, batch_pred_list, axis=0)
        pred_scores = np.append(pred_scores, batch_pred_scores, axis=0)
        user_list = np.append(user_list, user.numpy(), axis=0)

    result = []
    for user, pred, score in zip(user_list, pred_list, pred_scores):
        for idx, item in enumerate(pred):
            result.append(
                (
                    int(user_id2token[user]),
                    int(item_id2token[item]),
                    score[idx],
                )
            )

    print(f"generate submission file at {args.train.submit_dir}/{args.model}.csv")

    result = pd.DataFrame(result, columns = ["user", "item", "score"])
    result.drop(columns=['score'],inplace=True)
    
    result.to_csv(
        f"{args.root_dir}{args.train.submit_dir}/{args.model}.csv", index=False
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    arg = parser.add_argument
    arg("--config", 
        "-c",
        help='Configuration 파일을 설정합니다.',
        default='../configs/config.yaml'
    )
    arg("--root_dir",
        "-r",
        type=str,
        default='/data/ephemeral/home/'
    )
    arg("--model",
        "-m",
        type=str,
        help="name of recommender"
    )
    arg("--ckpt_file",
        "-ck",
        type=str,
        help='loads the checkpoint of trained model'
    )
    args = parser.parse_args()

    #### Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args에 있는 값이 config_yaml에 있는 값보다 우선함. (단, None이 아닌 값일 경우)
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]
    args = config_yaml

    #### set config and args
    config = Config(model=args.model, config_file_list=['../configs/recbole_model.yaml', '../configs/recbole_setting.yaml'])

    inference(args, config)