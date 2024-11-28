import os
import random
import numpy as np
import pickle
import torch
import time
import logging
import json
from tqdm import tqdm
import importlib
from collections import defaultdict
import torch.nn.functional as F

class Setting:
    @staticmethod
    def seed_everything(seed):
        """
        [description]
        seed 값을 고정시키는 함수입니다.

        [arguments]
        seed : seed 값
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def __init__(self):
        now = time.localtime()
        now_date = time.strftime("%Y%m%d", now)
        now_hour = time.strftime("%X", now)
        save_time = now_date + "_" + now_hour.replace(":", "")
        self.save_time = save_time

    def get_log_path(self, args):
        """
        [description]
        log file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        path : log file을 저장할 경로를 반환합니다.
        이 때, 경로는 log/날짜_시간_모델명/ 입니다.
        """
        path = os.path.join(args.train.log_dir, f"{self.save_time}_{args.model}/")
        self.make_dir(path)

        return path

    def get_submit_filename(self, args):
        """
        [description]
        submit file을 저장할 경로를 반환하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.

        [return]
        filename : submit file을 저장할 경로를 반환합니다.
        이 때, 파일명은 submit/날짜_시간_모델명.csv 입니다.
        """
        if not os.path.exists(args.train.submit_dir):
            os.makedirs(args.train.submit_dir)
        filename = os.path.join(
            args.train.submit_dir, f"{self.save_time}_{args.model}.csv"
        )
        return filename

    def make_dir(self, path):
        """
        [description]
        경로가 존재하지 않을 경우 해당 경로를 생성하며, 존재할 경우 pass를 하는 함수입니다.

        [arguments]
        path : 경로

        [return]
        path : 경로
        """
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            pass
        return path

    def model_modular(self, args, filename="dataset", fun_name=None):
        """
        [description]
        model의 data에 대한 module를 진행하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
        filename : 동적 모듈을 불러올 filename을 설정합니다.
        fun_name : 모듈에서 불러올 function을 설정합니다. (dafault : None)

        [return]
        output : 불러올 동적 모듈을 리턴합니다.
        """
        model_name = args.model
        model_folder = os.path.join(args.dataloader.data_path, model_name)
        module_path = os.path.join(model_folder, f"{filename.lower()}.py")

        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Module {filename} not found in {model_folder}.")

        module_name = f"{filename.lower()}"
        module = importlib.import_module(module_name)

        if fun_name:
            output = getattr(module, f"{fun_name}")
        elif fun_name == None:
            output = getattr(module, f"{model_name}{'Dataset'}")

        return output

    def save_file(self, args, data, file_extension=".npy", type=None):
        """
        [description]
        data를 지정된 경로로 저장하는 함수입니다.

        [arguments]
        args : argparse로 입력받은 args 값으로 이를 통해 모델의 정보를 전달받습니다.
        data : data 전달받습니다.
        file_extension : 저장할 파일의 확장자를 받습니다.(dafault: 'npy')
        """
        # 저장 경로 생성
        if not os.path.exists(args.train.predict_dir):
            os.makedirs(args.train.predict_dir)
        filename = os.path.join(
            args.train.predict_dir, f"{self.save_time}_{args.model}"
        )

        # 데이터 타입에 따라 처리
        if file_extension == ".npy":
            filename = os.path.join(
                args.train.predict_dir, f"{self.save_time}_{args.model}{file_extension}"
            )
            if isinstance(data, np.ndarray):
                np.save(filename, data)
            elif isinstance(data, torch.Tensor):
                np.save(filename, data.cpu().numpy())
            else:
                raise ValueError(
                    "지원하지 않는 데이터 형식입니다. NumPy 배열 또는 PyTorch 텐서만 허용됩니다."
                )
        elif file_extension == ".pkl":
            filename = os.path.join(
                args.train.predict_dir,
                f"{self.save_time}_{args.model}_" + type + f"{file_extension}",
            )
            with open(filename, "wb") as file:
                pickle.dump(data, file)

    def get_latest_file(self, args, directory, model, file_extension=".npy"):
        """
        [description]
        주어진 디렉토리에서 가장 최근 파일을 찾는 함수입니다.

        [arguments]
        directory (str): 파일들이 저장된 디렉토리 경로입니다.
        model : 찾고자 하는 파일의 모델을 전달받습니다.
        file_extension (str): 찾고자 하는 파일의 확장자입니다. (default: ".npy")
        """
        if args.ensemble_type == 'soft':
            directory = os.path.join(directory, model + "/predict/")
            
        files = [f for f in os.listdir(directory) if f.endswith(file_extension)]

        if not files:
            print("해당 디렉토리에 파일이 없습니다.")
            return None

        if file_extension == ".pkl":
            # 'user'와 'item' 파일을 따로 찾아 최신 파일을 각각 찾기
            user_files = [f for f in files if "user" in f]
            item_files = [f for f in files if "item" in f]

            # 'user'와 'item' 파일이 각각 존재하면 최신 파일을 찾음
            if user_files:
                latest_user_file = max(
                    user_files, key=lambda x: x.split("_")[0] + x.split("_")[1]
                )
            else:
                latest_user_file = None

            if item_files:
                latest_item_file = max(
                    item_files, key=lambda x: x.split("_")[0] + x.split("_")[1]
                )
            else:
                latest_item_file = None

            # 파일 경로 반환 (없으면 None 반환)
            latest_user_path = (
                os.path.join(file_path, latest_user_file) if latest_user_file else None
            )
            latest_item_path = (
                os.path.join(file_path, latest_item_file) if latest_item_file else None
            )

            with open(latest_user_path, "rb") as file:
                latest_user_dict = pickle.load(file)

            with open(latest_item_path, "rb") as file:
                latest_item_dict = pickle.load(file)

            return latest_user_dict, latest_item_dict

        elif file_extension == ".csv":
            # .csv 파일 처리: 각 모델 별 submit csv
            file_path = os.path.join(directory, model) + file_extension

            # latest_file = max(files, key=lambda x: x.split("_")[0] + x.split("_")[1])
            # latest_file_path = os.path.join(file_path, latest_file)
            # return latest_file_path
            return file_path

        else:
            # .npy 파일에 대한 처리
            latest_file = max(files, key=lambda x: x.split("_")[0] + x.split("_")[1])
            return os.path.join(file_path, latest_file)


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
        self.formatter = logging.Formatter("[%(asctime)s] - %(message)s")

        self.file_handler = logging.FileHandler(self.path + "train.log")
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def log(self, epoch, train_loss, valid_loss, valid_r10):
        """
        [description]
        log file에 epoch, train loss, valid loss를 기록하는 함수입니다.
        이 때, log file은 train.log로 저장됩니다.

        [arguments]
        epoch : epoch
        train_loss : train loss
        valid_loss : valid loss
        """
        message = f"epoch : {epoch}/{self.args.train.epochs} | train loss : {train_loss:.3f} | valid loss : {valid_loss:.3f} | valid_r10 : {valid_r10: .3f}"
        self.logger.info(message)

    def close(self):
        """
        [description]
        log file을 닫는 함수입니다.
        """
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def save_args(self):
        """
        [description]
        model에 사용된 args를 저장하는 함수입니다.
        이 때, 저장되는 파일명은 model.json으로 저장됩니다.
        """
        argparse_dict = self.args.__dict__
        if "_content" in argparse_dict:
            content_dict = argparse_dict["_content"]  # '_content' 부분만 추출
        else:
            raise KeyError(
                "'_content' key not found in args. Please check the structure of args."
            )

        # 직렬화 불가능한 객체를 걸러내기 위한 함수
        def _json_serializable(obj):
            try:
                json.dumps(obj)
                return True
            except TypeError:
                return False

        # 직렬화가 가능한 항목만 필터링
        content_dict_serializable = {
            key: value
            for key, value in content_dict.items()
            if _json_serializable(value)
        }

        # JSON 파일로 저장
        with open(f"{self.path}/model.json", "w") as f:
            json.dump(content_dict_serializable, f, indent=4)

    def __del__(self):
        self.close()


####################
# ensemble 전용 function
def optimize_replace_inf(multi_output):
    for i in range(multi_output.size(0)):
        mask = multi_output[i] != float("-inf")
        min_value = torch.min(multi_output[i][mask])

        multi_output[i][multi_output[i] == float("-inf")] = min_value

    return multi_output


def row_min_max_normalization(tensor):
    row_min = tensor.min(dim=1, keepdim=True).values
    row_max = tensor.max(dim=1, keepdim=True).values

    normalized_tensor = (tensor - row_min) / (row_max - row_min)

    return normalized_tensor


def transform_df_to_dict(data):
    data_dic = defaultdict(list)
    data_df = {}
    for u, i in tqdm(zip(data["user_idx"], data["item_idx"])):
        data_dic[u].append(i)

    for user in data_dic:
        data_df[user] = data_dic[user]

    return data_df


def get_total_probability(logit_list):
    proba = np.stack(logit_list)
    proba = torch.tensor(proba)
    proba[proba == float("inf")] = float("-inf")
    probabilities = F.softmax(proba, dim=1)

    # 확률값 내림차순으로 정렬
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    
    return sorted_probabilities