import sys
sys.path.append("../")

import pandas as pd
import os
import argparse
import yaml
from datetime import datetime
from pytz import timezone

from dataloader import *
from dataset import MakeMatrixDataSet
from utils import evaluate
from EASER import EASER

import warnings
warnings.filterwarnings(action="ignore")



def parse_args(config_path="src/EASER/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return argparse.Namespace(**config)



def main(args):
    data_loader = Dataloader(config = args)
    train_df, users, items = data_loader.dataloader()
    
    model = EASER(args)
    predict_save_path = os.path.join(args.output_dir, 'predictions.npy')  # 저장 경로 설정
    model.fit(train_df, predict_save_path=predict_save_path)

    # 결과 저장 (유저별 Top-K 추천)
    result_df = model.predict(train_df, users, items, args.train['K'])
    file_name = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d_%H%M%S')
    result_csv_path = os.path.join(args.output_dir, f"{file_name}_{args.train['K']}.csv")
    result_df[["user", "item"]].to_csv(result_csv_path, index=False)
    print(f"Recommendation results saved as CSV at {result_csv_path}")


   
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)