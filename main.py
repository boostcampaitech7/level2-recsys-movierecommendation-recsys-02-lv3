import subprocess
import argparse
import os
from omegaconf import OmegaConf


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='parser')

    arg = parser.add_argument
    str2dict = lambda x: {k:int(v) for k,v in (i.split(':') for i in x.split(','))}
    
    
    arg('--config', '-c', '--c', type=str, 
        help='Configuration 파일을 설정합니다.', default='choi/level2-recsys-movierecommendation-recsys-02-lv3/configs/config.yaml')    
    arg('--model', '-m', '--m', type=str, 
            choices=['MultiVAE', 'EASE', 'EASER', 'MF', 'BERT4Rec'],
            help='학습 및 예측할 모델을 선택할 수 있습니다.')
    args = parser.parse_args()
    
    ######################## Config with yaml  
    config_args = OmegaConf.create(vars(args))
 
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # 선택된 모델 이름을 기반으로 model_names 불러오기
    if args.model and args.model in config_yaml.model_names:
        config_yaml.model_names = {"name": config_yaml.model_names[args.model]["name"]}
    else:
        raise ValueError(f"Invalid model name: {args.model}")
    
    
    
    model_name = config_yaml.model_names['name']
    print("-"*15 + f"{model_name} File Load" + "-"*15)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        subprocess.run(
            ["python", os.path.join(current_dir, f"src/{model_name}/main.py")], check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {model_name}/main.py: {e}")
