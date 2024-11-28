import argparse
import os
from tqdm import tqdm
from collections import Counter
import pandas as pd
from omegaconf import OmegaConf
from src.utils.util import Setting

def ranking_priority_sort(user, model_results, selected_items, top_k):
    for _ in range(10):
        for model_idx, model_result in enumerate(model_results):
            model_items = model_result[model_result["user"] == user]["item"].tolist()
            leftover_items = [item for item in model_items if item not in selected_items]
            if _ >= len(leftover_items):
                # 교집합 아이템을 제외한 아이템이 더 없을 경우, 다음 모델의 결과값으로 넘어갑니다.
                continue
            selected_items.append(leftover_items[_])
            
        if len(selected_items) >= 10:
            selected_items = selected_items[:top_k]
            return selected_items

def model_priority_sort(user, model_results, selected_items, num_list, top_k):
    for model_idx, model_result in enumerate(model_results):
        model_items = model_result[model_result["user"] == user]["item"].tolist() # 현재 모델 추천 아이템 
        additional_items = [item for item in model_items if item not in selected_items] # 선택된 아이템 중 교집합과 겹치지 않는 아이템
        selected_items.extend(additional_items[:num_list[model_idx]])
    return selected_items[:top_k]

def main(args):

    ##### Setting
    Setting.seed_everything(args.seed)
    setting = Setting()
    
    model_list = args.models
    top_k = 10  # 최종 추천 아이템 개수

    # 모델별 최신 CSV 파일 경로 정의
    model_csv_paths = [
        setting.get_latest_file(args, args.submit_dir, model, file_extension=".csv") for model in model_list
    ]

    print(model_csv_paths)
    # None 값 제거
    model_csv_paths = [path for path in model_csv_paths if path is not None]
    if not model_csv_paths:
        print("Error: 모델 CSV 파일 경로를 찾을 수 없습니다.")
        return
    
    # 모델별 CSV 파일 불러오기
    model_results = [pd.read_csv(path) for path in model_csv_paths]
    
    # 유저별로 추천 아이템을 리스트로 변환
    user_recommendations = {}
    for model_idx, model_result in enumerate(model_results):
        for user, group in model_result.groupby("user"):
            if user not in user_recommendations:
                user_recommendations[user] = []
            # 모델의 추천 아이템을 리스트에 추가
            user_recommendations[user].extend(group["item"].tolist())

    # 하드보팅 앙상블
    final_recommendations = []
    for user, items in tqdm(user_recommendations.items(), desc="Processing users"):
        item_counter = Counter(items) # 아이템 등장 빈도 계산
        
        selected_items = [item for item, count in item_counter.items() if count >= 2] # 교집합(2번 이상 등장 아이템)

        if args.sort == 'rank':
            selected_items = ranking_priority_sort(user, model_results, selected_items, top_k)
        elif args.sort == 'model':
            num_list = list(map(int, args.ensemble_nums))  # 각 모델별 사용 개수 리스트 ex)[5, 3, 2]
            selected_items = model_priority_sort(user, model_results, selected_items, num_list, top_k)

        for item in selected_items:
            final_recommendations.append((user, item)) # 최종 추천에 추가

    print('-'*10, f'Submission', '-'*10)
    result_df = pd.DataFrame(final_recommendations, columns=["user", "item"])


    print('-'*10, f'Save', '-'*10)
    file_path = setting.make_dir(os.path.join(args.submit_dir, "ensemble"))
    models_combined = ",".join(model_list)
    filename = os.path.join(file_path, f"ensemble_{args.sort}({models_combined}).csv")
    result_df.to_csv(filename, index=False)

    print('Done!')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parser')
    
    arg = parser.add_argument  
    str2dict = lambda x: {k: int(v) for k, v in (i.split(':') for i in x.split(','))}
    
    arg('--config', '-c', '--c', type=str,  
        help='Configuration 파일을 설정합니다.', default='configs/config.yaml')    
    arg('--models', '-m', '--m', nargs="+", required=True,
        help='앙상블에 사용할 모델 list를 설정합니다.') 
    arg('--seed', '-s', '--s', type=int, default=0,
        help='데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.')
    arg('--ensemble_type', '-et', default='hard')
    arg('--sort', '-sort', choices=['rrf', 'rank', 'model'], default='rank',
        help='교집합을 제외한 추천 아이템의 순서를 정렬할 방법을 선택합니다.')
    arg('--ensemble_nums', '-ew', nargs="+",
        help='model sort 방식을 이용할 때 ensemble에 사용할 각 모델별 num list를 반환합니다.(model과 동일한 순서로 설정)')
    

    args = parser.parse_args()
    
    config_args = OmegaConf.create(vars(args))  
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    del config_yaml.model_names
    
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    print(OmegaConf.to_yaml(config_yaml))

    main(config_yaml) 
    
#####
# directory = '/data/ephemeral/home/yusol/level2-recsys-movierecommendation-recsys-02-lv3/saved/ensemble'
# models = ['BERT4Rec', 'EASER']
# take_counts = [5, 5]
# top_k = 10

# model_csv_paths = [
#     get_latest_file(directory, model, file_extension=".csv") for model in models
# ]