import argparse
import yaml

from recbole.config import Config
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from dataloader import get_data

import torch
import numpy as np
import pandas as pd

def inference(args, config):
    tr_data, val_data, te_data = get_data(config)
    (
        inference_config,
        model,
        inference_dataset,
        _,
        _,
        _,
    ) = load_data_and_model(args.checkpoint_path)
    
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
    pred_list = batch_pred_list.clone().detach().cpu().numpy()
    pred_scores = batch_pred_scores.clone().detach().cpu().numpy()
    user_list = user.numpy()

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

    print(f"generate submission file at {args.submit_dir}{args.model}.csv")
    result = pd.DataFrame(result, columns = ["user", "item", "score"])
    result.drop(columns=['score'],inplace=True)
    result.to_csv(
        f"{args.submit_dir}{args.model}.csv", index=False
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--model",
        "-m",
        type=str,
        help="name of recommender"
    )
    arg("--checkpoint_path",
        "-c",
        type=str,
        help='loads the checkpoint of trained model'
    )
    arg("--submit_dir",
        "-s",
        type=str,
        default="./submit/"
    )
    args = parser.parse_args()
    config = Config(model=args.model, config_file_list=['Config/run.yaml', 'Config/setting.yaml'])

    inference(args, config)
