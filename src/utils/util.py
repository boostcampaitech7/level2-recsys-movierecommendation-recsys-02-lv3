import os
import random
import numpy as np
import torch
import time
from src.models import MultiVAE
import logging
import json
import importlib


class Setting:
    @staticmethod
    def seed_everything(seed):
        '''
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        '''
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self):
        now = time.localtime()
        now_date = time.strftime('%Y%m%d', now)
        now_hour = time.strftime('%X', now)
        save_time = now_date + '_' + now_hour.replace(':', '')
        self.save_time = save_time
    
    def get_log_path(self, args):
        '''
        [description]
        log file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        path : log file을 저장할 경로를 반환합니다.
        이 때, 경로는 log/날짜_시간_모델명/ 입니다.
        '''
        path = os.path.join(args.train.log_dir, f'{self.save_time}_{args.model}/')
        self.make_dir(path)
        
        return path

    def get_submit_filename(self, args):
        '''
        [description]
        submit file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        filename : submit file을 저장할 경로를 반환합니다.
        이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
        '''
        if not os.path.exists(args.train.submit_dir):
            os.makedirs(args.train.submit_dir)
        filename = os.path.join(args.train.submit_dir, f'{self.save_time}_{args.model}.csv')
        return filename
    
    def make_dir(self,path):
        '''
        [description]
        경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

        [arguments]
        path : 경로

        [return]
        path : 경로
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path
    
    
    def model_modular(self, args, filename = 'dataset', fun_name = None):
        model_name = args.model
        model_folder = os.path.join(args.dataloader.data_path, model_name)
        module_path = os.path.join(model_folder, f"{model_name}_{filename.lower()}.py")
        
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Module {filename} not found in {model_folder}.")
        
        module_name = f"src.data.{model_name}.{model_name}_{filename.lower()}"
        module = importlib.import_module(module_name)
        
        if fun_name:
            output = getattr(module, f'{fun_name}')
        elif fun_name == None:
            output = getattr(module, f"{model_name}{'Dataset'}")
        
        return output
    
    

class Logger:
    def __init__(self, args, path):
        """
        [description]
        log file을 생성하는 클래스입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
        path : log file을 저장할 경로를 전달받습니다.
        """
        self.args = args
        self.path = path

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('[%(asctime)s] - %(message)s')

        self.file_handler = logging.FileHandler(self.path+'train.log')
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, epoch, train_loss, valid_loss, valid_r10):
        '''
        [description]
        log file에 epoch, train loss, valid loss를 기록하는 함수입니다.
        이 때, log file은 train.log로 저장됩니다.

        [arguments]
        epoch : epoch
        train_loss : train loss
        valid_loss : valid loss
        '''
        message = f'epoch : {epoch}/{self.args.train.epochs} | train loss : {train_loss:.3f} | valid loss : {valid_loss:.3f} | valid_r10 : {valid_r10: .3f}'
        self.logger.info(message)
    
    def close(self):
        '''
        [description]
        log file을 닫는 함수입니다.
        '''
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def save_args(self):
        '''
        [description]
        model에 사용된 args를 저장하는 함수입니다.
        이 때, 저장되는 파일명은 model.json으로 저장됩니다.
        '''
        argparse_dict = self.args.__dict__
        if '_content' in argparse_dict:
            content_dict = argparse_dict['_content']  # '_content' 부분만 추출
        else:
            raise KeyError("'_content' key not found in args. Please check the structure of args.")
        
        # 직렬화 불가능한 객체를 걸러내기 위한 함수
        def _json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except TypeError:
                return False

        # 직렬화가 가능한 항목만 필터링
        content_dict_serializable = {key: value for key, value in content_dict.items() if _json_serializable(value)}

        # JSON 파일로 저장
        with open(f'{self.path}/model.json', 'w') as f:
            json.dump(content_dict_serializable, f, indent=4)

    def __del__(self):
        self.close()