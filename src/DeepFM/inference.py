from tqdm import tqdm
import torch
from dataset import make_inference_data, inference_mapping


def deepfm_predict(args, model, data, idx_dict):
    """
    DeepFM 모델을 사용하여 각 사용자별 Top-K 추천 아이템을 추출하는 함수

    parameters
    ----------
    model : train 이후 best epoch의 저장된 모델 이용
    data : user, item의 상호작용이 담긴 데이터 프레임
    idx_dict : 각 컬럼을 mapping하기 위한 dict

    returns
    -------
    predict_output : 예측 결과
    """
    model.eval()

    # Inference
    print("make inference data...")
    LAST_USER_ID = len(data["user"].unique()) - 1
    USER = 0
    ITEM = 1
    SCORE = 2

    batch_size = args.test.batch_size
    user_list = [i * batch_size for i in range(1, LAST_USER_ID // batch_size)]
    user_list.append(LAST_USER_ID)
    slice_start = 0
    predict_output = []

    data["user"] = data["user"].map(idx_dict["user2idx"])
    for slice_end in tqdm(user_list):
        temp_list = [i for i in range(slice_start, slice_end + 1)]
        sliced_df = data.query("user in @temp_list")
        inference_df = make_inference_data(sliced_df)
        inference_data = inference_mapping(inference_df, idx_dict, args)
        predict_data = model(inference_data)
        predict_data = torch.cat(
            [inference_data[:, 0:2], predict_data.unsqueeze(1)], dim=1
        )  # make top_k list

        # predict_data.int()
        for user in temp_list:
            temp_data = predict_data[predict_data[:, USER] == user][:, SCORE]
            topk = torch.topk(temp_data, 10, sorted=True)
            for item in predict_data[topk.indices, ITEM]:
                predict_output.append((user, int(item)))
        slice_start = slice_end + 1

    return predict_output
